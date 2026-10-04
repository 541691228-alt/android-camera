package cn.yege.dshcam

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

/**
 * FocusAssist 的 JVM 单元测试：只依赖 java.nio.ByteBuffer，不碰任何 android.* 类。
 */
class FocusAssistTest {

    /** 构造一个单平面（pixelStride = 1）的灰度缓冲。 */
    private fun plane(width: Int, height: Int, value: (x: Int, y: Int) -> Byte): ByteBuffer {
        val buf = ByteBuffer.allocate(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                buf.put(value(x, y))
            }
        }
        buf.rewind()
        return buf
    }

    /** 1) 均匀平面：亮度全 100，格子内没有对比度，grad 必须全 0。 */
    @Test
    fun buildGridUniformPlane() {
        val w = 32
        val h = 24
        val buf = plane(w, h) { _, _ -> 100.toByte() }
        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = 32, pixelStride = 1, cell = 8)

        assertEquals(4, grid.cols)   // ceil(32 / 8)
        assertEquals(3, grid.rows)   // ceil(24 / 8)
        assertEquals(12, grid.luma.size)
        assertEquals(12, grid.grad.size)

        for (i in grid.luma.indices) {
            assertEquals(100, grid.luma[i])
            assertEquals(0, grid.grad[i])
        }
        // at / gradAt 的寻址也要与行优先一致
        assertEquals(100, grid.at(3, 2))
        assertEquals(0, grid.gradAt(3, 2))
    }

    /** 2) 行末 padding：rowStride > width，右侧的 255 填充不能进入任何格子。 */
    @Test
    fun buildGridHandlesRowPadding() {
        val w = 32
        val h = 24
        val rowStride = 40
        val buf = ByteBuffer.allocate(rowStride * h)
        for (y in 0 until h) {
            for (x in 0 until rowStride) {
                buf.put(if (x < w) 100.toByte() else 255.toByte())
            }
        }
        buf.rewind()

        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = rowStride, pixelStride = 1, cell = 8)
        assertEquals(4, grid.cols)
        assertEquals(3, grid.rows)

        for (i in grid.luma.indices) {
            assertEquals(100, grid.luma[i])
        }
    }

    /** 3) pixelStride = 2：只有偶数绝对下标是有效亮度，奇数位置是丢弃字节。 */
    @Test
    fun buildGridHandlesPixelStride() {
        val w = 32
        val h = 24
        val rowStride = 64
        val buf = ByteBuffer.allocate(rowStride * h)
        for (y in 0 until h) {
            val rowStart = y * rowStride
            for (x in 0 until w) {
                val idx = rowStart + x * 2
                buf.put(idx, if (idx % 2 == 0) 100.toByte() else 0.toByte())
            }
        }
        buf.rewind()

        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = rowStride, pixelStride = 2, cell = 8)
        assertEquals(4, grid.cols)
        assertEquals(3, grid.rows)

        for (i in grid.luma.indices) {
            assertEquals(100, grid.luma[i])
        }
    }

    /** 4) 斑马纹：只有亮度达到阈值的格子被标记。 */
    @Test
    fun zebraMaskFlagsBrightCells() {
        val w = 32
        val h = 8
        // x 属于 [8, 16) 的一整列填 250（对应 cell=8 时的第 2 格）
        val buf = plane(w, h) { x, _ -> if (x in 8 until 16) 250.toByte() else 100.toByte() }

        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = w, pixelStride = 1, cell = 8)
        assertEquals(4, grid.cols)
        assertEquals(1, grid.rows)

        val mask = FocusAssist.zebraMask(grid)
        assertEquals(4, mask.size)
        for (i in mask.indices) {
            if (i == 1) assertTrue(mask[i]) else assertFalse(mask[i])
        }
    }

    /** 5) 平坦区域不应该产生峰值对焦提示。 */
    @Test
    fun peakingMaskIgnoresFlatArea() {
        val w = 32
        val h = 24
        val buf = plane(w, h) { _, _ -> 100.toByte() }

        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = w, pixelStride = 1, cell = 8)
        val mask = FocusAssist.peakingMask(grid)

        assertEquals(grid.luma.size, mask.size)
        for (b in mask) {
            assertFalse(b)
        }
    }

    /** 6) 2 像素一组的棋盘有强梯度，亮度足够，应被大量标记。 */
    @Test
    fun peakingMaskFlagsDetail() {
        val w = 32
        val h = 24
        // x/2 为偶数的整列填 0，为奇数的整列填 255 → 平均亮度约 127
        val buf = plane(w, h) { x, _ -> if ((x / 2) % 2 == 0) 0.toByte() else 255.toByte() }

        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = w, pixelStride = 1, cell = 8)
        val mask = FocusAssist.peakingMask(grid)

        var trueCount = 0
        for (b in mask) {
            if (b) trueCount++
        }
        assertTrue("至少应有 cols 个格子被标记，实际 $trueCount", trueCount >= grid.cols)
    }

    /** 7) 同样的棋盘但整体太暗，低于 minLuma 阈值时不得标记。 */
    @Test
    fun peakingMaskRespectsMinLuma() {
        val w = 32
        val h = 24
        // 两档值 0 / 30，平均亮度约 15，低于 PEAK_MIN_LUMA = 20
        val buf = plane(w, h) { x, _ -> if ((x / 2) % 2 == 0) 0.toByte() else 30.toByte() }

        val grid = FocusAssist.buildGrid(buf, w, h, rowStride = w, pixelStride = 1, cell = 8)
        val mask = FocusAssist.peakingMask(grid)

        assertEquals(grid.luma.size, mask.size)
        for (b in mask) {
            assertFalse(b)
        }
    }

    /** 8) 退化输入：不抛异常，返回 1×1 的全 0 网格。 */
    @Test
    fun buildGridDegenerateInput() {
        val buf = ByteBuffer.allocate(4)
        buf.rewind()

        val grid = FocusAssist.buildGrid(buf, 0, 0, rowStride = 0, pixelStride = 1, cell = 8)

        assertEquals(1, grid.cols)
        assertEquals(1, grid.rows)
        assertEquals(1, grid.luma.size)
        assertEquals(1, grid.grad.size)
        assertEquals(0, grid.luma[0])
        assertEquals(0, grid.grad[0])
    }
}
