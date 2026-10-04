package cn.yege.dshcam

/**
 * 主体模型（u2netp）的纯数学部分。
 *
 * 不 import 任何 android.*，所以可以在普通 JVM 单元测试里跑。
 * 模型规格（离线实测确认）：
 *  - 输入 `[1,3,320,320]` float，RGB 通道优先（NCHW），像素 /255 后按 ImageNet 均值方差归一化；
 *    缩放是**直接拉伸**到 320×320（官方 u2netp 推理就是这么做的，不改宽高比）。
 *  - 输出 `[1,1,320,320]` 显著度（U-2-Net 的 d0 融合图），值域每次都不一样，得先归一化。
 */
object SaliencyMath {

    /** 模型输入边长 */
    const val SIDE = 320

    /**
     * 网格峰值可信门槛：低于它认为"画面里没有主体"。
     *
     * 离线实测（用户实拍的黑色鼠标 + 红垫）：鼠标那张降采样到 64×48 后峰值 0.996；
     * 同一模型看空无一物的纯红鼠标垫，峰值只有 0.573（只有左下角渐晕的一点杂点）。
     * 取 0.75 能把两者分开 —— 这正是用户抱怨的"没主体也一直催"的解药。
     */
    const val GRID_PEAK_GATE = 0.75f

    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * side×side 的 ARGB 像素 → NCHW float（3*side*side，R 通道整块在前）。
     * 与 Python 端 `_u2net_test.py` 的预处理逐位对齐（已验证模型能抠出主体）。
     */
    fun argbToNchw(argb: IntArray, side: Int): FloatArray {
        val n = side * side
        val out = FloatArray(3 * n)
        for (i in 0 until n) {
            val c = argb[i]
            val r = ((c shr 16) and 0xFF) / 255f
            val g = ((c shr 8) and 0xFF) / 255f
            val b = (c and 0xFF) / 255f
            out[i] = (r - MEAN[0]) / STD[0]
            out[n + i] = (g - MEAN[1]) / STD[1]
            out[2 * n + i] = (b - MEAN[2]) / STD[2]
        }
        return out
    }

    /**
     * side×side 的原始掩膜 → cols×rows 网格（0..1，行优先）：
     * 先按最小/最大值归一化（模型输出值域不固定），再按格子面积平均（盒式降采样，
     * 比"取格子中心一个像素"稳，避免小杂点被放大成一个可信主体）。
     */
    fun maskToGrid(mask: FloatArray, side: Int, cols: Int, rows: Int): FloatArray {
        var mn = Float.MAX_VALUE
        var mx = -Float.MAX_VALUE
        for (v in mask) {
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        val range = mx - mn
        val stretch = range > 1e-6f
        val out = FloatArray(cols * rows)
        for (r in 0 until rows) {
            val y0 = r * side / rows
            val y1 = maxOf(y0 + 1, (r + 1) * side / rows)
            for (c in 0 until cols) {
                val x0 = c * side / cols
                val x1 = maxOf(x0 + 1, (c + 1) * side / cols)
                var sum = 0f
                var cnt = 0
                for (y in y0 until y1) {
                    val base = y * side
                    for (x in x0 until x1) {
                        val v = mask[base + x]
                        sum += if (stretch) (v - mn) / range else 0f
                        cnt++
                    }
                }
                out[r * cols + c] = if (cnt > 0) sum / cnt else 0f
            }
        }
        return out
    }

    /** 网格最大值（用 [GRID_PEAK_GATE] 判"有没有主体"） */
    fun peak(grid: FloatArray): Float {
        var m = 0f
        for (v in grid) if (v > m) m = v
        return m
    }

    /**
     * 显示方向归一化坐标 (u,v) → 传感器方向归一化坐标，写进 out[0]/out[1]。
     *
     * 实时引导拿到的帧是**传感器方向**的（YUV），而预览和 `OverlayView` 用的是转过
     * `rotationDegrees` 的显示方向；要让"主体在哪"这件事对得上，降采样时就得按下面
     * 这套逆映射取源像素（与 `FaceAnalyzer.rotateGrid` 用的是同一套公式，有单测守着）：
     *   90°（顺时针）→ su = v,     sv = 1-u
     *   270°（逆时针）→ su = 1-v,   sv = u
     *   180° → su = 1-u, sv = 1-v
     *   其它 → 原样
     * @param out 长度 ≥2 的复用数组（逐像素调用，避免每次分配）
     */
    fun displayToSensor(degrees: Int, u: Float, v: Float, out: FloatArray) {
        val d = ((degrees % 360) + 360) % 360
        when (d) {
            90 -> {
                out[0] = v
                out[1] = 1f - u
            }
            180 -> {
                out[0] = 1f - u
                out[1] = 1f - v
            }
            270 -> {
                out[0] = 1f - v
                out[1] = u
            }
            else -> {
                out[0] = u
                out[1] = v
            }
        }
    }
}

/**
 * ★ 2026-10-05：实时引导的「模型锚点」。
 *
 * 用户选了「模型当 1–2 秒一次的锚点、帧间仍用规则网格」这条路：
 * 每 [FaceAnalyzer.ANCHOR_PERIOD_MS] 把一帧降采样成 320×320 喂 u2netp，
 * 得到"这一帧里到底有没有主体、主体在哪"的权威答案；帧间那些来不及跑模型的帧，
 * 继续用 `AutoFrame.salienceFromGrid` 的规则网格跟位置。
 *
 * @param x/y 主体质心（显示方向归一化 0..1；没有主体时是 0.5/0.5 占位）
 * @param hasSubject [peak] 是否过 [SaliencyMath.GRID_PEAK_GATE]（没过＝画面里没有主体）
 * @param atMs 生成时刻（`SystemClock.elapsedRealtime()`），用来判"锚点还新不新"
 */
data class SubjectAnchor(
    val x: Float,
    val y: Float,
    val strength: Float,
    val spread: Float,
    val peak: Float,
    val atMs: Long,
    val hasSubject: Boolean
)
