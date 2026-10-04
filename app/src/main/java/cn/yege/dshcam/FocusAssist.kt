package cn.yege.dshcam

import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.abs
import kotlin.math.max

/**
 * 峰值对焦 + 斑马纹计算核心。
 *
 * 输入是相机 YUV 帧的 Y（亮度）平面，输出为按 cell 缩小的格子网格，
 * 上层覆盖层据此在合焦（高对比度）区域画亮点、在过曝区域画斑马纹。
 *
 * 本文件为纯 Kotlin：只依赖 java.nio.ByteBuffer 与 kotlin.math，
 * 不引用任何 android.* 类，因此可直接在 JVM 单元测试中运行。
 */
object FocusAssist {

    /** 默认格子边长（像素）。 */
    const val CELL = 12

    /** 斑马纹亮度阈值：>= 235 视为过曝。 */
    const val ZEBRA_LUMA = 235

    /** 峰值对焦梯度阈值：格子平均局部对比度 >= 18 视为可能合焦。 */
    const val PEAK_GRAD = 18

    /** 峰值对焦最低亮度：过暗区域不参与峰值提示，避免噪点误判。 */
    const val PEAK_MIN_LUMA = 20

    /**
     * 缩小的格子网格。
     *
     * @param cols 列数
     * @param rows 行数
     * @param luma 每格平均亮度，长度 cols*rows，行优先
     * @param grad 每格平均局部对比度，长度 cols*rows，行优先
     */
    data class AssistGrid(val cols: Int, val rows: Int, val luma: IntArray, val grad: IntArray) {
        fun at(col: Int, row: Int): Int = luma[row * cols + col]
        fun gradAt(col: Int, row: Int): Int = grad[row * cols + col]
    }

    /**
     * 由 Y 平面构建缩小的亮度/对比度网格。
     *
     * @param buffer      含 Y 平面的缓冲区（可能是整帧数据，width/height 描述其有效区域）
     * @param width       Y 平面宽度（像素）
     * @param height      Y 平面高度（像素）
     * @param rowStride   行字节跨度
     * @param pixelStride 相邻像素字节跨度
     * @param cell        格子边长
     * @return 网格；参数非法时返回 1×1 全 0 网格
     */
    fun buildGrid(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        cell: Int = CELL
    ): AssistGrid {
        // 参数非法或缓冲区为空：返回 1×1 全 0 网格，绝不抛异常
        if (width <= 0 || height <= 0 || cell <= 0 || rowStride <= 0 || pixelStride <= 0) {
            return AssistGrid(1, 1, intArrayOf(0), intArrayOf(0))
        }

        val cols = max(1, ceil(width.toDouble() / cell.toDouble()).toInt())
        val rows = max(1, ceil(height.toDouble() / cell.toDouble()).toInt())
        val count = cols * rows

        val luma = IntArray(count)
        val grad = IntArray(count)

        // 每格内部采样步长：最多 4×4 = 16 个采样点
        val step = max(1, cell / 4)

        // 一律使用绝对下标读取，因此先把 position 归零，避免受调用方游标影响
        buffer.rewind()
        val limit = buffer.limit()

        var row = 0
        while (row < rows) {
            val y0 = row * cell
            var col = 0
            while (col < cols) {
                val x0 = col * cell
                val yEnd = minOf(y0 + cell, height)
                val xEnd = minOf(x0 + cell, width)

                var sum = 0L
                var n = 0
                var gradSum = 0L
                var gradN = 0

                var y = y0
                while (y < yEnd) {
                    var x = x0
                    while (x < xEnd) {
                        // 绝对下标：idx = 行*y 方向的 rowStride + 列*x 方向的 pixelStride
                        // 这样即使 buffer 带 offset（例如整帧里的 Y 平面切片）也始终命中正确字节。
                        val idx = y * rowStride + x * pixelStride
                        if (idx < 0 || idx + 1 > limit) {
                            x += step
                            continue
                        }
                        val center = buffer.get(idx).toInt() and 0xFF
                        sum += center
                        n++

                        var best = 0
                        // 邻居按 step 距离取：在 4×4 稀疏采样下，相邻 1 像素的差分
                        // 会大量被噪声主导、且细纹理几乎不可分辨；按 step 取正好跨过
                        // 一个采样间隔，衡量的是"真实存在的结构边缘"的对比度，
                        // 且与中心点的采样密度一致，不会重复统计同一个像素。
                        val rx = x + step
                        if (rx < xEnd) {
                            val ridx = y * rowStride + rx * pixelStride
                            if (ridx >= 0 && ridx + 1 <= limit) {
                                val rv = buffer.get(ridx).toInt() and 0xFF
                                best = abs(center - rv)
                            }
                        }
                        val by = y + step
                        if (by < yEnd) {
                            val bidx = by * rowStride + x * pixelStride
                            if (bidx >= 0 && bidx + 1 <= limit) {
                                val bv = buffer.get(bidx).toInt() and 0xFF
                                val d = abs(center - bv)
                                if (d > best) best = d
                            }
                        }

                        if (rx < xEnd || by < yEnd) {
                            gradSum += best
                            gradN++
                        }

                        x += step
                    }
                    y += step
                }

                val i = row * cols + col
                luma[i] = if (n > 0) ((sum + n / 2) / n).toInt() else 0
                // 格内只有一个采样点时无法形成邻居对，grad 记 0
                grad[i] = if (gradN > 0) ((gradSum + gradN / 2) / gradN).toInt() else 0

                col++
            }
            row++
        }

        val available = buffer.remainingBytes()
        if (available < 0) {
            return AssistGrid(1, 1, intArrayOf(0), intArrayOf(0))
        }

        return AssistGrid(cols, rows, luma, grad)
    }

    /** 过曝斑马纹掩码：luma[i] >= threshold。 */
    fun zebraMask(grid: AssistGrid, threshold: Int = ZEBRA_LUMA): BooleanArray {
        val n = grid.luma.size
        val out = BooleanArray(n)
        var i = 0
        while (i < n) {
            out[i] = grid.luma[i] >= threshold
            i++
        }
        return out
    }

    /** 峰值对焦掩码：亮度足够且局部对比度足够。 */
    fun peakingMask(
        grid: AssistGrid,
        grad: Int = PEAK_GRAD,
        minLuma: Int = PEAK_MIN_LUMA
    ): BooleanArray {
        val n = grid.luma.size
        val out = BooleanArray(n)
        var i = 0
        while (i < n) {
            out[i] = grid.luma[i] >= minLuma && grid.grad[i] >= grad
            i++
        }
        return out
    }

    /** 计算缓冲区当前可读字节数（不改变 position）。 */
    private fun ByteBuffer.remainingBytes(): Int {
        val pos = position()
        val lim = limit()
        val r = lim - pos
        position(pos)
        return r
    }
}
