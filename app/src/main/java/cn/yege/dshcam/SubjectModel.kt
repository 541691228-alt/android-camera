package cn.yege.dshcam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer

/**
 * ★ 2026-10-05 新增：主体模型（u2netp，U-2-Net small，显著目标检测）。
 *
 * 为什么要它：自写的规则算法（`AutoFrame.salienceFromArgb`）在用户实拍场景里认不出主体 ——
 * 「黑色鼠标放在红色鼠标垫上」那张，规则版 55% 的格子超过重要阈值、质心被背景拽到画面正中，
 * 于是自动构图怎么裁都不加分。换成中心-周边对比后好转（spread 0.551 → 0.033），但在
 * 「画面里根本没有主体」时仍会给出假响应。u2netp 离线实测（桌面 CPU 67 ms）：
 * 鼠标那张掩膜就是整只鼠标（64×48 网格峰值 0.996），纯红垫那张峰值只有 0.573
 * —— 它自己就能回答"这里到底有没有主体"，这正是实时引导不再乱催的关键。
 *
 * 体积与代价：模型 4.4 MB 打进 assets；ONNX Runtime AAR 约 26.6 MB（4 个 ABI）。
 * 加载/推理都在调用线程（拍照链路本来就在后台），失败一律返回 null 由规则算法兜底。
 */
object SubjectModel {

    private const val TAG = "DSHCam"
    private const val ASSET = "u2netp.onnx"

    @Volatile
    private var env: OrtEnvironment? = null

    @Volatile
    private var session: OrtSession? = null

    /** 最近一次推理的摘要（给日志用）："峰值=0.996 耗时=280ms" */
    @Volatile
    var lastInfo: String = ""
        private set

    /** 最近一次网格峰值（0..1），低于 [SaliencyMath.GRID_PEAK_GATE] 就是"没有主体" */
    @Volatile
    var lastPeak: Float = 0f
        private set

    fun isReady(): Boolean = session != null

    /**
     * 加载模型（幂等）。第一次会读 assets 4.4 MB + 建 ONNX 会话，约几百毫秒到 1 秒。
     * 返回 false 表示不可用，调用方应回退到规则算法。
     */
    @Synchronized
    fun ensureLoaded(context: Context): Boolean {
        if (session != null) return true
        val t0 = System.currentTimeMillis()
        return try {
            val bytes = context.assets.open(ASSET).use { it.readBytes() }
            val e = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setInterOpNumThreads(1)
            }
            val s = e.createSession(bytes, opts)
            env = e
            session = s
            Log.d(TAG, "主体模型：加载成功（${bytes.size / 1024} KB，${System.currentTimeMillis() - t0} ms）")
            true
        } catch (t: Throwable) {
            Log.w(TAG, "主体模型：加载失败，回退到规则算法（${t.localizedMessage}）")
            false
        }
    }

    /**
     * 跑一次推理。传入任意尺寸位图（内部会拉伸到 320×320，跟离线验证脚本一致）。
     *
     * 返回 `[1,1,320,320]` 的 d0 融合图（原始值，未归一化）；失败返回 null。
     * 归一化/降采样交给 [SaliencyMath.maskToGrid]（纯 Kotlin，有单测）。
     */
    fun runMask(bitmap: Bitmap): FloatArray? {
        var small: Bitmap? = null
        return try {
            val side = SaliencyMath.SIDE
            small = Bitmap.createScaledBitmap(bitmap, side, side, true)
            val argb = IntArray(side * side)
            small.getPixels(argb, 0, side, 0, 0, side, side)
            runMaskFromArgb(argb, side)
        } catch (t: Throwable) {
            Log.w(TAG, "主体模型：推理失败，回退到规则算法（${t.localizedMessage}）")
            null
        } finally {
            if (small != null && small !== bitmap) {
                try {
                    small.recycle()
                } catch (_: Throwable) {
                }
            }
        }
    }

    /**
     * ★ 2026-10-05：直接吃 ARGB 像素的版本（实时引导用）。
     *
     * 实时那条路每 1.5 秒才跑一次，帧是 YUV、不需要再建 Bitmap —— 调用方
     * （`FaceAnalyzer`）自己按显示方向降采样成 320×320 的 ARGB 交进来就行，
     * 省掉一次大图缩放和一次 Bitmap 分配。
     */
    fun runMaskFromArgb(argb: IntArray, side: Int): FloatArray? {
        val e = env ?: return null
        val s = session ?: return null
        if (argb.size < side * side) return null
        return try {
            val t0 = System.currentTimeMillis()
            val nchw = SaliencyMath.argbToNchw(argb, side)
            val buf = FloatBuffer.allocate(nchw.size)
            buf.put(nchw)
            buf.rewind()
            val shape = longArrayOf(1, 3, side.toLong(), side.toLong())
            OnnxTensor.createTensor(e, buf, shape).use { tensor ->
                val inputName = s.inputNames.first()
                s.run(mapOf(inputName to tensor)).use { res ->
                    @Suppress("UNCHECKED_CAST")
                    val plane = (res.get(0).value as Array<Array<Array<FloatArray>>>)[0][0]
                    val out = FloatArray(plane.size * plane[0].size)
                    var k = 0
                    for (row in plane) for (v in row) out[k++] = v
                    lastInfo = "耗时=${System.currentTimeMillis() - t0}ms"
                    out
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "主体模型：推理失败，回退到规则算法（${t.localizedMessage}）")
            null
        }
    }

    /**
     * 一站式：位图 → cols×rows 显著度网格（0..1）。
     *
     * 同时更新 [lastPeak]（网格峰值，供调用方判"有没有主体"）。失败返回 null。
     */
    fun salienceGrid(bitmap: Bitmap, cols: Int, rows: Int): FloatArray? {
        val mask = runMask(bitmap) ?: return null
        return finishGrid(mask, cols, rows)
    }

    /** 一站式：ARGB 像素（side×side）→ cols×rows 显著度网格，机制同 [salienceGrid] */
    fun salienceGridFromArgb(argb: IntArray, side: Int, cols: Int, rows: Int): FloatArray? {
        val mask = runMaskFromArgb(argb, side) ?: return null
        return finishGrid(mask, cols, rows)
    }

    private fun finishGrid(mask: FloatArray, cols: Int, rows: Int): FloatArray {
        val grid = SaliencyMath.maskToGrid(mask, SaliencyMath.SIDE, cols, rows)
        lastPeak = SaliencyMath.peak(grid)
        lastInfo = lastInfo + " 峰值=${"%.3f".format(lastPeak)}"
        return grid
    }
}
