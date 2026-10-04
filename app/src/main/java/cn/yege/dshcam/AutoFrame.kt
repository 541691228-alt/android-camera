package cn.yege.dshcam

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * AI 自动构图算法核心。
 *
 * 纯 Kotlin 实现，不依赖任何 android.* 类，可以直接在 JVM 单元测试里跑。
 * 两个用途：
 *  (A) 实时引导：预览时从亮度/梯度网格算出主体位置，告诉用户镜头该往哪挪；
 *  (B) 拍完自动裁：从缩略图的显著性图里搜出最佳裁剪框，把主体放到三分线上。
 */
object AutoFrame {

    /**
     * 最近一次 bestCrop 的评分摘要（"采纳裁剪/保持最大框 最佳=x 基准=y"）。
     * 纯数据、不依赖 android，方便 CameraController 打日志、也方便排查"为什么不裁"。
     */
    var lastScoreInfo: String = ""

    /** 显著性（重点）图：cols*rows 个 0..1 权重，行优先 */
    data class Salience(val cols: Int, val rows: Int, val weight: FloatArray) {
        fun at(col: Int, row: Int): Float = weight[row * cols + col]
    }

    /** 主体：归一化位置（0..1） + 强度 + 占画面比例 */
    data class Subject(val x: Float, val y: Float, val strength: Float, val spread: Float)

    /** 目标裁剪框（源图像素坐标） */
    data class CropRect(val left: Int, val top: Int, val width: Int, val height: Int, val score: Float)

    /**
     * 构图引导。
     *
     * [confident] = 这一帧有没有找到"可信主体"。false 时不画十字/箭头，只在卡片里给一句提示，
     * 避免把"弥散的显著性质心"当主体、没完没了地催人挪镜头。
     * [framed] = 主体已经进了三分点容差（构图这一项达标）；[ok] = framed 且水平也端平、
     * 可以直接按快门。
     */
    data class Guide(val text: String, val targetX: Float, val targetY: Float,
                     val dx: Float, val dy: Float, val ok: Boolean,
                     val confident: Boolean = true, val framed: Boolean = false)

    /** 三分点常量 */
    private const val ONE_THIRD = 1f / 3f
    private const val TWO_THIRDS = 2f / 3f

    /** 肤色判定阈值：满足条件时认为该像素很可能是人脸/皮肤，给显著性加权 */
    private fun isSkin(r: Int, g: Int, b: Int): Boolean {
        val mx = max(r, max(g, b))
        val mn = min(r, min(g, b))
        return r > 95 && g > 40 && b > 20 && r > g && r > b && (mx - mn) > 15
    }

    /**
     * 把 ARGB 图缩成 cols×rows 的显著性图。
     *
     * 算法＝「中心-周边对比」（Itti 风格的简化版，见 [centerSurround]）+ 人脸肤色加成。
     *
     * 为什么不用"和全图平均色/平均亮度的差"（我先后试了两版，都被真机照片否掉，记录在此）：
     *  - 只统计边缘/肤色/饱和度：用户实拍的「黑色鼠标放在红色鼠标垫上」判不出主体 ——
     *    整幅平均权重被垫子抬平，55% 的格子都超过"重要"阈值，质心被拽到画面正中；
     *  - 再加一项"与全图主色调的色差"：还是不行。那张照片的背景不是纯色，而是带很强
     *    明暗渐变的红垫（右上亮、左下暗），"离平均色远"于是把四周暗角也算成了主体
     *    （离线重放：max 0.407→0.325，质心仍停在 0.421、spread 0.551，鼠标照样被淹没）。
     * 中心-周边对付渐变天然有效：平滑的渐变在模糊前后一样，差值≈0；只有"跟周围不一样"
     * 的主体才留下响应。两个尺度（半径 2 / 6）累加，小细节和整块主体都能抓到。
     */
    fun salienceFromArgb(argb: IntArray, width: Int, height: Int, cols: Int, rows: Int): Salience {
        // 参数非法时返回全 0 的空图，绝不抛异常
        if (width <= 0 || height <= 0 || cols <= 0 || rows <= 0 || argb.size < width * height) {
            val n = max(0, cols) * max(0, rows)
            return Salience(max(0, cols), max(0, rows), FloatArray(n))
        }
        val n = cols * rows
        val out = FloatArray(n)
        // 每格平均色：算中心-周边对比要用
        val cellR = FloatArray(n)
        val cellG = FloatArray(n)
        val cellB = FloatArray(n)
        for (row in 0 until rows) {
            val y0 = row * height / rows
            val y1 = max(y0 + 1, (row + 1) * height / rows)
            for (col in 0 until cols) {
                val x0 = col * width / cols
                val x1 = max(x0 + 1, (col + 1) * width / cols)

                var sumEdge = 0.0
                var sumSkin = 0.0
                var sumR = 0.0
                var sumG = 0.0
                var sumB = 0.0
                var count = 0

                var y = y0
                while (y < y1 && y < height) {
                    var x = x0
                    while (x < x1 && x < width) {
                        val v = argb[y * width + x]
                        val r = (v ushr 16) and 0xFF
                        val g = (v ushr 8) and 0xFF
                        val b = v and 0xFF
                        sumR += r
                        sumG += g
                        sumB += b

                        val luma = (299 * r + 587 * g + 114 * b) / 1000.0

                        // 与右边、下边像素的亮度差；越界方向记 0
                        var edge = 0.0
                        if (x + 1 < width) {
                            val vr = argb[y * width + x + 1]
                            val lr = ((vr ushr 16) and 0xFF)
                            val lg = ((vr ushr 8) and 0xFF)
                            val lb = (vr and 0xFF)
                            val lumaR = (299 * lr + 587 * lg + 114 * lb) / 1000.0
                            edge += abs(luma - lumaR)
                        }
                        if (y + 1 < height) {
                            val vd = argb[(y + 1) * width + x]
                            val dr = ((vd ushr 16) and 0xFF)
                            val dg = ((vd ushr 8) and 0xFF)
                            val db = (vd and 0xFF)
                            val lumaD = (299 * dr + 587 * dg + 114 * db) / 1000.0
                            edge += abs(luma - lumaD)
                        }
                        sumEdge += (edge / 2.0 / 255.0)

                        if (isSkin(r, g, b)) sumSkin += 1.0

                        count++
                        x++
                    }
                    y++
                }

                val cnt = if (count > 0) count.toDouble() else 1.0
                val e = sumEdge / cnt
                val k = sumSkin / cnt
                val idx = row * cols + col
                // 绝对项只留"边缘"和"肤色"：饱和度是绝对量（一整块纯红垫子每格都一样高），
                // 留着会给背景铺一层均匀底座，正是"质心被拽到画面中间"的老毛病，故删。
                out[idx] = (0.70 * e + 0.30 * k).toFloat().coerceIn(0f, 1f)
                cellR[idx] = (sumR / cnt).toFloat()
                cellG[idx] = (sumG / cnt).toFloat()
                cellB[idx] = (sumB / cnt).toFloat()
            }
        }

        // 中心-周边对比：主判据（0.80），边缘/肤色项为辅助（0.20）
        val cs = centerSurround(cellR, cellG, cellB, cols, rows)
        for (i in 0 until n) {
            out[i] = (0.80f * cs[i] + 0.20f * out[i]).coerceIn(0f, 1f)
        }
        return Salience(cols, rows, out)
    }

    /**
     * 中心-周边对比（Itti 1998 那套的简化版）：每个格子跟自己"模糊后的自己"比。
     *
     * 用亮度 + 两个色彩对立通道（红-绿、蓝-黄），在半径 2 和 6 两个尺度上累加
     * |原图 − 模糊|，最后按最大值归一化到 0..1。
     *  - 平滑渐变（渐晕、墙面明暗、纯色大背景）：模糊前后几乎一样 → 响应≈0，不干扰主体；
     *  - 主体（跟周围明显不同的一块）：响应大 → 质心落在主体上。
     */
    private fun centerSurround(cellR: FloatArray, cellG: FloatArray, cellB: FloatArray, cols: Int, rows: Int): FloatArray {
        val n = cols * rows
        val out = FloatArray(n)
        if (n <= 0 || cellR.size < n || cellG.size < n || cellB.size < n) return out
        val l = FloatArray(n)
        val rg = FloatArray(n)
        val by = FloatArray(n)
        for (i in 0 until n) {
            val r = cellR[i]
            val g = cellG[i]
            val b = cellB[i]
            l[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            rg[i] = (r - g) / 255f
            by[i] = (b - (r + g) * 0.5f) / 255f
        }
        for (radius in intArrayOf(2, 6)) {
            val bl = boxBlur(l, cols, rows, radius)
            val brg = boxBlur(rg, cols, rows, radius)
            val bby = boxBlur(by, cols, rows, radius)
            for (i in 0 until n) {
                out[i] += abs(l[i] - bl[i]) + abs(rg[i] - brg[i]) + abs(by[i] - bby[i])
            }
        }
        // 噪声地板：浮点残差、传感器噪点经过"按最大值归一化"会被放大成满格 1.0
        // （纯色图本该处处为 0，实测会出现 1e-7 量级的残差）。低于地板的一律当 0。
        val floor = 1e-4f
        var maxV = 0f
        for (i in 0 until n) {
            if (out[i] < floor) out[i] = 0f
            if (out[i] > maxV) maxV = out[i]
        }
        if (maxV > 0f) for (i in 0 until n) out[i] = (out[i] / maxV).coerceIn(0f, 1f)
        return out
    }

    /** 可分离盒式模糊（边缘钳位），"周边"就是它；半径越大＝比的范围越大 */
    private fun boxBlur(src: FloatArray, cols: Int, rows: Int, radius: Int): FloatArray {
        val n = cols * rows
        if (n <= 0 || src.size < n || radius <= 0 || cols <= 0 || rows <= 0) return src.copyOf(minOf(src.size, maxOf(n, 0)))
        val tmp = FloatArray(n)
        val dst = FloatArray(n)
        val win = (2 * radius + 1).toFloat()
        for (row in 0 until rows) {
            val base = row * cols
            var sum = 0f
            for (i in -radius..radius) sum += src[base + i.coerceIn(0, cols - 1)]
            for (col in 0 until cols) {
                tmp[base + col] = sum / win
                sum += src[base + (col + radius + 1).coerceIn(0, cols - 1)] -
                    src[base + (col - radius).coerceIn(0, cols - 1)]
            }
        }
        for (col in 0 until cols) {
            var sum = 0f
            for (i in -radius..radius) sum += tmp[i.coerceIn(0, rows - 1) * cols + col]
            for (row in 0 until rows) {
                dst[row * cols + col] = sum / win
                sum += tmp[(row + radius + 1).coerceIn(0, rows - 1) * cols + col] -
                    tmp[(row - radius).coerceIn(0, rows - 1) * cols + col]
            }
        }
        return dst
    }

    /**
     * 按 EXIF 方向把显著性网格转正（转到「用户看到的方向」）。
     *
     * 为什么需要：竖着拿手机拍（EXIF 6/8）时，传感器其实是横的，`BitmapFactory` 解出来的
     * 图也是横的，而用户看到、相册里存的却是竖的。原来的自动构图直接在传感器坐标系里判构图，
     * 于是「主体明明偏在画面右边」在横过来的坐标系里可能正好落在三分点上 → 判「已经达标」、
     * 不裁（用户实拍的黑色鼠标照片就是这样，重新对齐后它在传感器坐标系里 x≈0.34）。
     * 6 = 顺时针 90° 才正；8 = 逆时针 90°；3 = 180°；其余原样返回。
     */
    fun rotateSalience(s: Salience, orientation: Int): Salience {
        if (orientation != 6 && orientation != 8 && orientation != 3) return s
        val c = s.cols
        val r = s.rows
        if (c <= 0 || r <= 0 || s.weight.size < c * r) return s
        if (orientation == 3) {
            val out = FloatArray(c * r)
            for (row in 0 until r) {
                for (col in 0 until c) {
                    out[(r - 1 - row) * c + (c - 1 - col)] = s.weight[row * c + col]
                }
            }
            return Salience(c, r, out)
        }
        // 转 90°：转正后的列数 = 原来的行数
        val nc = r
        val nr = c
        val out = FloatArray(nc * nr)
        for (row in 0 until r) {
            for (col in 0 until c) {
                val w = s.weight[row * c + col]
                if (orientation == 6) {
                    // 顺时针 90°：传感器 (col,row) → 显示 (r-1-row, col)
                    out[col * nc + (r - 1 - row)] = w
                } else {
                    // 逆时针 90°：传感器 (col,row) → 显示 (row, c-1-col)
                    out[(c - 1 - col) * nc + row] = w
                }
            }
        }
        return Salience(nc, nr, out)
    }

    /**
     * 把「显示方向」里算出来的裁剪框映射回传感器方向的像素坐标。
     *
     * 原图是按传感器方向解的、也只能按那个坐标系裁（EXIF 方向标签原样写给新文件），
     * 所以 90° 的两档要交换宽高。
     */
    fun mapCropToSensor(crop: CropRect, sensorW: Int, sensorH: Int, orientation: Int): CropRect {
        if (sensorW <= 0 || sensorH <= 0) return crop
        val l: Int
        val t: Int
        val w: Int
        val h: Int
        when (orientation) {
            6 -> {
                l = crop.top
                t = sensorH - crop.left - crop.width
                w = crop.height
                h = crop.width
            }
            8 -> {
                l = sensorW - crop.top - crop.height
                t = crop.left
                w = crop.height
                h = crop.width
            }
            3 -> {
                l = sensorW - crop.left - crop.width
                t = sensorH - crop.top - crop.height
                w = crop.width
                h = crop.height
            }
            else -> {
                l = crop.left
                t = crop.top
                w = crop.width
                h = crop.height
            }
        }
        val left = l.coerceIn(0, max(0, sensorW - 1))
        val top = t.coerceIn(0, max(0, sensorH - 1))
        val width = w.coerceIn(1, max(1, sensorW - left))
        val height = h.coerceIn(1, max(1, sensorH - top))
        return CropRect(left, top, width, height, crop.score)
    }

    /**
     * 用亮度网格 + 梯度网格算显著性（给实时引导用，避免每帧读整帧像素）。
     * 梯度占比更大：构图引导关心的是「轮廓/主体在哪」，而不是整幅画面的绝对亮度。
     */
    fun salienceFromGrid(luma: IntArray, grad: IntArray, cols: Int, rows: Int): Salience {
        val n = max(0, cols) * max(0, rows)
        val out = FloatArray(n)
        if (cols <= 0 || rows <= 0) return Salience(max(0, cols), max(0, rows), out)
        // ★ 2026-10-04 修：亮度项要减去整幅平均亮度。
        //   原来 w = 0.7*grad + 0.3*luma，平整又明亮的大背景（例如一整块红色鼠标垫）
        //   每格都能拿到同样的基础权重，"主体"被背景淹掉、质心永远落在画面中间，
        //   于是引导一会儿说"抬高一点"、一会儿说"靠近一点"，永远不收敛（用户反馈）。
        //   减掉均值后只剩"比周围亮/暗"的部分，平整画面权重≈0 → 直接判"没找到主体"。
        var sumL = 0.0
        for (i in 0 until minOf(n, luma.size)) sumL += luma[i].toDouble()
        val meanL = if (n > 0 && luma.isNotEmpty()) (sumL / minOf(n, luma.size)).toFloat() else 0f
        for (i in 0 until n) {
            val l = if (i < luma.size) luma[i].toFloat() else 0f
            val g = if (i < grad.size) grad[i].toFloat() else 0f
            val lDev = ((l - meanL) / 128f).coerceIn(-1f, 1f)
            val w = 0.7f * (g / 64f) + 0.3f * max(0f, lDev)
            out[i] = w.coerceIn(0f, 1f)
        }
        return Salience(cols, rows, out)
    }

    /**
     * 用加权质心定位主体。
     *
     * 为什么用加权质心而不是「取权重最大的那一格」：
     * 单个最大格很容易被噪点、反光点或一小块高对比纹理带偏；加权质心把整块主体区域
     * 的贡献平均起来，得到的落点更稳、更贴近主体的真实重心，抖动也更小。
     * 同时只统计 weight >= 0.15*maxWeight 的格子，过滤掉大片背景噪声，避免背景把质心拖走。
     */
    fun subjectCenter(s: Salience): Subject? {
        if (s.cols <= 0 || s.rows <= 0) return null
        if (s.weight.size < s.cols * s.rows) return null

        var maxWeight = 0f
        for (w in s.weight) if (w > maxWeight) maxWeight = w
        if (maxWeight <= 0f) return null

        val threshold = 0.15f * maxWeight
        val strongThreshold = 0.5f * maxWeight

        var sumW = 0.0
        var sumX = 0.0
        var sumY = 0.0
        var strongCount = 0

        for (row in 0 until s.rows) {
            for (col in 0 until s.cols) {
                val w = s.weight[row * s.cols + col]
                if (w >= strongThreshold) strongCount++
                if (w >= threshold) {
                    val wd = w.toDouble()
                    sumW += wd
                    sumX += wd * (col + 0.5)
                    sumY += wd * (row + 0.5)
                }
            }
        }

        if (sumW <= 0.0) return null

        val x = (sumX / sumW / s.cols).toFloat().coerceIn(0f, 1f)
        val y = (sumY / sumW / s.rows).toFloat().coerceIn(0f, 1f)
        val total = (s.cols * s.rows).toDouble()
        val strength = (sumW / total).toFloat().coerceIn(0f, 1f)
        val spread = (strongCount / total).toFloat().coerceIn(0f, 1f)

        return Subject(x, y, strength, spread)
    }

    /**
     * 构图引导。
     *
     * 关于平移方向：dx = 目标三分点 - 主体当前位置，是「主体还需要往哪走」。
     * 画面里的内容要往右走，镜头就得往左平移（内容在取景框里随相机反向移动），
     * 所以 dx > 0 时提示「镜头往左移一点」；垂直方向同理，dy > 0 说明主体偏上，
     * 需要让内容下移，镜头抬高一点即可让画面内容整体下移，故提示「镜头抬高一点」。
     *
     * 2026-10-04 用户反馈："永远是向右一点向左一点靠近一点，没一个度吗？"
     * 根因：原来不管主体可不可信，都拿显著度的加权质心当主体。画面里没有明确主体时
     * （显著度铺满整幅），质心落在画面中间、离最近的三分点永远超过 5%，于是提示永远在催。
     * 现在三种结果：
     *   ① 没找到可信主体（强度太弱 / 显著度铺满整幅）→ confident=false：不画箭头十字，
     *      只给一句"对准要拍的东西"，不再无休止催人挪镜头；
     *   ② 有主体但没进容差 → 给方向 + 还差多少（百分数，就是用户要的那个"度"）；
     *   ③ 已经进容差 → "构图 OK"，并且带迟滞：达标之后要偏出更大的 0.11 才重新催，
     *      免得在容差边缘来回横跳。
     *
     * @param settled 上一帧是否已经"构图 OK"（由调用方保存），用来实现上面的迟滞。
     * @param trusted 上一帧是否已经把当前主体当成"可信主体"。用来给强度门限做迟滞：
     *   真机日志（19:32）里规则的强度在 0.039 / 0.069 之间抖，只用一个 0.06 的门限就会让文案在
     *   「镜头往左移一点（约 13%）」和「没找到明显主体」之间来回跳 —— 同一个"没度"的毛病，
     *   只是换了副面孔。所以：没被信任过要 ≥0.06 才认，信任之后掉到 0.045 以下才收回。
     */
    fun guide(
        subject: Subject?,
        levelOk: Boolean,
        settled: Boolean = false,
        trusted: Boolean = false,
    ): Guide {
        if (subject == null) {
            return Guide("把主体放进画面", ONE_THIRD, ONE_THIRD, 0f, 0f, false, false)
        }

        val x = subject.x.coerceIn(0f, 1f)
        val y = subject.y.coerceIn(0f, 1f)
        val targetX = if (abs(x - ONE_THIRD) <= abs(x - TWO_THIRDS)) ONE_THIRD else TWO_THIRDS
        val targetY = if (abs(y - ONE_THIRD) <= abs(y - TWO_THIRDS)) ONE_THIRD else TWO_THIRDS

        // ① 没找到可信主体：显著度太弱、铺满整幅（spread 大 = 到处都是"重点"），
        //    或者相反 —— 亮点少得可怜（spread 极小且强度也低），那基本是噪点不是主体
        val strengthGate = if (trusted) 0.045f else 0.06f
        val noSubject = subject.strength < strengthGate ||
            subject.spread > 0.40f ||
            (subject.spread < 0.02f && subject.strength < 0.25f)
        if (noSubject) {
            return Guide("没找到明显主体，对准要拍的东西", targetX, targetY, 0f, 0f, false, false, false)
        }

        val dx = targetX - x
        val dy = targetY - y

        // ③ 容差（迟滞）：没到位时 6%，达标后放宽到 11%
        val tol = if (settled) 0.11f else 0.06f
        if (abs(dx) <= tol && abs(dy) <= tol) {
            val text = if (levelOk) "构图 OK，可以拍" else "构图 OK，先端平手机"
            return Guide(text, targetX, targetY, dx, dy, levelOk, true, true)
        }

        // ② 给出方向 + 距离（百分数＝画面宽/高的百分之多少）
        val text: String
        if (subject.spread < 0.02f) {
            text = "靠近一点"
        } else if (abs(dx) >= abs(dy)) {
            val pct = Math.round(abs(dx) * 100f)
            text = if (dx > 0f) "镜头往左移一点（约 $pct%）" else "镜头往右移一点（约 $pct%）"
        } else {
            val pct = Math.round(abs(dy) * 100f)
            text = if (dy > 0f) "镜头抬高一点（约 $pct%）" else "镜头压低一点（约 $pct%）"
        }
        return Guide(text, targetX, targetY, dx, dy, false, true, false)
    }

    /**
     * 在 srcW×srcH 的源图里搜最佳裁剪框。
     *
     * 搜索策略：先按目标宽高比取最大可用框，再按面积线性生成 21 档缩放尺寸，
     * 每档用固定步长（1/16 框宽）扫位置，逐个打分取最优。
     * 这样既能保证主体尽量落在三分交点、又尽量不把重要的高权重区域裁掉，
     * 同时偏好面积大的框（避免为了构图把画面裁得太碎）。
     *
     * edgeCut 的用意：框外那些高权重格子（>= 0.6*maxWeight）大概率是主体的一部分，
     * 把它们裁到画面外是构图大忌，所以这两倍权重的重罚会强力排斥「切到主体」的候选框。
     */
    fun bestCrop(s: Salience, srcW: Int, srcH: Int, aspect: Float, minKeep: Float = 0.70f): CropRect {
        // 1) 最大可用框
        val safeAspect = if (aspect.isFinite() && aspect > 0f) aspect else 1f
        val maxW: Int
        val maxH: Int
        if (srcW.toFloat() / srcH.toFloat() > safeAspect) {
            maxH = srcH
            maxW = (srcH * safeAspect).roundToInt()
        } else {
            maxW = srcW
            maxH = (srcW / safeAspect).roundToInt()
        }
        val maxWc = maxW.coerceIn(1, max(1, srcW))
        val maxHc = maxH.coerceIn(1, max(1, srcH))

        // 居中兜底框（参数非法或全 0 时返回它）
        val fallback = CropRect(
            left = ((srcW - maxWc) / 2).coerceIn(0, max(0, srcW - maxWc)),
            top = ((srcH - maxHc) / 2).coerceIn(0, max(0, srcH - maxHc)),
            width = maxWc,
            height = maxHc,
            score = 0f
        )

        // 参数校验：非法直接返回兜底，绝不抛异常
        if (s.cols <= 0 || s.rows <= 0 || srcW <= 0 || srcH <= 0) return fallback
        if (s.weight.size < s.cols * s.rows) return fallback

        // ★ 2026-10-05 修：maxWc/maxHc 是「源图像素」，而下面候选框 w_k/h_k 是「网格格子数」，
        //   原来直接把像素数当格子数用（又被 coerceAtMost 夹到 cols/rows），导致请求的宽高比
        //   彻底失效 —— 单测 bestCropKeepsSubjectInsideAndHonorsAspect 要 1:1 却拿到 4:3
        //   就是这个坑。这里换算成格子数（至少 1 格），后面统一用格子口径。
        val maxWg = max(1, (maxWc.toDouble() * s.cols / srcW).roundToInt()).coerceAtMost(s.cols)
        val maxHg = max(1, (maxHc.toDouble() * s.rows / srcH).roundToInt()).coerceAtMost(s.rows)

        // 全部权重之和；全 0 时 coverage 记 1.0
        var totalWeight = 0.0
        var maxWeight = 0f
        for (i in 0 until s.cols * s.rows) {
            val w = s.weight[i]
            if (w > 0f) totalWeight += w.toDouble()
            if (w > maxWeight) maxWeight = w
        }
        if (totalWeight <= 0.0 || maxWeight <= 0f) return fallback

        val keep = minKeep.coerceIn(0.1f, 1f)
        val highThreshold = 0.6f * maxWeight

        // ★ 2026-10-05 新增：重要格子（权重 >= highThreshold）的总权重。
        //   用来把"被裁掉的重要显著度"归一化成比例，见 scoreCrop 末尾的说明。
        var highTotal = 0.0
        for (i in 0 until s.cols * s.rows) {
            val w = s.weight[i]
            if (w >= highThreshold) highTotal += w.toDouble()
        }

        var bestScore = Float.NEGATIVE_INFINITY
        var bestX = 0
        var bestY = 0
        var bestWk = 1
        var bestHk = 1

        // ★ 2026-10-05：k=0 是"最大可用框"（4:3 时就是整图），单独记一份当基准
        var baseScore = Float.NEGATIVE_INFINITY
        var baseX = 0
        var baseY = 0
        var baseWk = maxWg
        var baseHk = maxHg

        // 2) 21 档尺寸候选：按面积线性缩小，保证 w_k/h_k 仍约等于 aspect
        for (k in 0..20) {
            val scale = 1f - k * (1f - keep) / 20f
            val sScale = sqrt(scale.coerceAtLeast(0f))
            val wk = max(1, (maxWg * sScale).roundToInt()).coerceAtMost(s.cols)
            val hk = max(1, (maxHg * sScale).roundToInt()).coerceAtMost(s.rows)

            val stepX = max(1, wk / 16)
            val stepY = max(1, hk / 16)
            val maxX = max(0, s.cols - wk)
            val maxY = max(0, s.rows - hk)

            var yk = 0
            while (yk <= maxY) {
                var xk = 0
                while (xk <= maxX) {
                    val score = scoreCrop(s, xk, yk, wk, hk, totalWeight, maxWeight, highThreshold, highTotal, maxWg, maxHg)
                    if (k == 0 && score > baseScore) {
                        baseScore = score
                        baseX = xk
                        baseY = yk
                        baseWk = wk
                        baseHk = hk
                    }
                    if (score > bestScore) {
                        bestScore = score
                        bestX = xk
                        bestY = yk
                        bestWk = wk
                        bestHk = hk
                    }
                    if (xk == maxX) break
                    xk = min(maxX, xk + stepX)
                }
                if (yk == maxY) break
                yk = min(maxY, yk + stepY)
            }
        }

        if (bestScore == Float.NEGATIVE_INFINITY) return fallback

        // ★ 2026-10-05 加"动剪刀门槛"：裁剪必须比最大框（4:3 时＝整图）高出 0.05 分才采纳，
        //   否则原样返回最大框 —— 避免为了零点几分的抖动把照片裁小。返回最大框时，
        //   下游 CameraController 会判"构图已达标、不另存"，等于什么都不做（安全）。
        val base = if (baseScore == Float.NEGATIVE_INFINITY) 0f else baseScore
        val adopted = bestScore >= base + 0.05f
        if (!adopted) {
            bestX = baseX
            bestY = baseY
            bestWk = baseWk
            bestHk = baseHk
            bestScore = base
        }
        lastScoreInfo = (if (adopted) "采纳裁剪" else "保持最大框") +
            " 最佳=" + ((bestScore * 1000f).roundToInt() / 1000f) +
            " 基准=" + ((base * 1000f).roundToInt() / 1000f)

        // 5) 换算成源图像素并强制夹进画面
        val left = (bestX.toDouble() * srcW / s.cols).roundToInt()
        val top = (bestY.toDouble() * srcH / s.rows).roundToInt()
        val outW = (bestWk.toDouble() * srcW / s.cols).roundToInt().coerceAtLeast(1)
        val outH = (bestHk.toDouble() * srcH / s.rows).roundToInt().coerceAtLeast(1)

        val leftC = left.coerceIn(0, max(0, srcW - 1))
        val topC = top.coerceIn(0, max(0, srcH - 1))
        val widthC = outW.coerceIn(1, max(1, srcW - leftC))
        val heightC = outH.coerceIn(1, max(1, srcH - topC))

        return CropRect(leftC, topC, widthC, heightC, bestScore)
    }

    /**
     * 给单个候选框打分。
     * coverage/thirds 用框内权重统计，areaRatio 奖励大框，edgeCut 惩罚把高权重区域裁到框外。
     */
    private fun scoreCrop(
        s: Salience,
        xk: Int,
        yk: Int,
        wk: Int,
        hk: Int,
        totalWeight: Double,
        maxWeight: Float,
        highThreshold: Float,
        highTotal: Double,
        maxW: Int,
        maxH: Int
    ): Float {
        var inWeight = 0.0
        var inHigh = 0.0
        var sumW = 0.0
        var sumX = 0.0
        var sumY = 0.0
        // ★ 2026-10-05 新增：只累计「重要格子」（权重 >= highThreshold）的加权质心。
        //   原来看的是框内"全部"格子的质心，一大片低权重背景会把质心拽回框中间，
        //   于是"把主体挪到三分点"在分数上体现不出来 —— 实测鼠标偏在右边的照片
        //   （主体只占一小块、周围全是低权重鼠标垫）永远判"不用裁"。
        var sumHW = 0.0
        var sumHX = 0.0
        var sumHY = 0.0

        // 框内：格子中心点 (col+0.5, row+0.5) 落在 [xk, xk+wk) × [yk, yk+hk) 内
        for (row in yk until (yk + hk)) {
            if (row < 0 || row >= s.rows) continue
            for (col in xk until (xk + wk)) {
                if (col < 0 || col >= s.cols) continue
                val w = s.weight[row * s.cols + col]
                if (w <= 0f) continue
                val wd = w.toDouble()
                inWeight += wd
                sumW += wd
                sumX += wd * (col + 0.5)
                sumY += wd * (row + 0.5)
                if (w >= highThreshold) {
                    inHigh += wd
                    sumHW += wd
                    sumHX += wd * (col + 0.5)
                    sumHY += wd * (row + 0.5)
                }
            }
        }

        // ① 重要显著度保留率 = 框内重要格子权重 / 全图重要格子权重。
        //    这里必须用"重要格子"而不是"全部格子"：后者的分母含一大片低权背景，
        //    只要有背景被裁掉就掉分，结果"什么都不裁"永远最高分（真机实测过）。
        val retention = if (highTotal > 0.0) {
            (inHigh / highTotal).toFloat().coerceIn(0f, 1f)
        } else {
            (inWeight / totalWeight).toFloat().coerceIn(0f, 1f)
        }

        // 三分法得分：优先用「重要格子」的加权质心（即主体重心），
        // 一个重要格子都没有时才退回全部格子的质心
        var thirds = 0f
        val cw = if (sumHW > 0.0) sumHW else sumW
        val cxRaw = if (sumHW > 0.0) sumHX else sumX
        val cyRaw = if (sumHW > 0.0) sumHY else sumY
        if (cw > 0.0) {
            val cx = ((cxRaw / cw - xk) / wk.toDouble()).toFloat().coerceIn(0f, 1f)
            val cy = ((cyRaw / cw - yk) / hk.toDouble()).toFloat().coerceIn(0f, 1f)
            val dx1 = cx - ONE_THIRD
            val dy1 = cy - ONE_THIRD
            val dx2 = cx - TWO_THIRDS
            val dy2 = cy - TWO_THIRDS
            val d1 = sqrt((dx1 * dx1 + dy1 * dy1).toDouble())
            val d2 = sqrt((dx2 * dx1 * 0f + dx2 * dx2 + dy2 * dy2).toDouble())
            val nearest = min(d1, d2)
            thirds = (1.0 - nearest / 0.45).toFloat().coerceIn(0f, 1f)
        }

        val areaRatio = if (maxW > 0 && maxH > 0) {
            (wk.toFloat() * hk.toFloat() / (maxW.toFloat() * maxH.toFloat())).coerceIn(0f, 1f)
        } else 0f

        // ★ 2026-10-05 重写评分（原来那版的三个坑，全是真机实测踩出来的）：
        //   坑1 惩罚项用"权重绝对和"（几十上百）→ 任何比整图小的框都被罚穿，等于永不裁剪；
        //      现在改成保留率（0~1）派生，主体保留 75% 以上就不罚，低于才按缺口重罚。
        //   坑2 "覆盖度"用全部格子权重占比 → 分母被一大片低权背景撑大，不裁就是 1.0，
        //      等于系统性地奖励"什么都不做"；现在分母只算重要格子（retention）。
        //   坑3 三分法用框内全部格子的质心 → 背景把质心拽回框中心，"主体是否落在三分点"
        //      根本看不出来；现在用重要格子的质心（主体重心）。
        //   权重：三分 0.55 / 主体保留 0.25 / 面积 0.20。三分是主角（否则白算），
        //   面积仍有分但不再压过构图，主体被裁才重罚。
        val cutPenalty = if (retention < 0.75f) (0.75f - retention) * 1.6f else 0f

        return (0.55f * thirds + 0.25f * retention + 0.20f * areaRatio - cutPenalty)
    }
}
