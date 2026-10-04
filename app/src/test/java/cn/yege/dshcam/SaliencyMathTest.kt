package cn.yege.dshcam

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 主体模型（u2netp）纯数学部分的单测：预处理布局/归一化、掩膜降采样、以及
 * 「有没有主体」的峰值门槛。
 *
 * 这些数字不是随手编的 —— 预处理与离线 Python 验证脚本（`_u2net_test.py`，实测能把
 * 用户实拍的黑色鼠标从红垫里完整抠出来）逐位对齐；门槛来自两张真机照片的实测峰值
 * （鼠标 0.996 / 纯红垫 0.573）。
 */
class SaliencyMathTest {

    /** 布局：R 通道整块在前、G 次之、B 最后；数值 = (v/255 - mean) / std（ImageNet 归一化） */
    @Test
    fun argbToNchwLayoutAndNormalization() {
        val side = 2
        val argb = intArrayOf(
            0xFFFF0000.toInt(), 0xFF00FF00.toInt(),
            0xFF0000FF.toInt(), 0xFFFFFFFF.toInt()
        )
        val out = SaliencyMath.argbToNchw(argb, side)
        val n = side * side
        assertEquals(3 * n, out.size)

        // 像素 0 是纯红：R=1、G=0、B=0
        assertEquals((1f - 0.485f) / 0.229f, out[0], 1e-5f)
        assertEquals((0f - 0.456f) / 0.224f, out[n], 1e-5f)
        assertEquals((0f - 0.406f) / 0.225f, out[2 * n], 1e-5f)
        // 像素 3 是纯白：三通道都是 (1-mean)/std
        assertEquals((1f - 0.485f) / 0.229f, out[3], 1e-5f)
        assertEquals((1f - 0.456f) / 0.224f, out[n + 3], 1e-5f)
        assertEquals((1f - 0.406f) / 0.225f, out[2 * n + 3], 1e-5f)
        // 像素 2 是纯蓝：B 通道拉满
        assertEquals((1f - 0.406f) / 0.225f, out[2 * n + 2], 1e-5f)
    }

    /** 掩膜降采样：先按 min/max 拉满，再按格子面积平均 */
    @Test
    fun maskToGridNormalizesAndAverages() {
        val side = 4
        val mask = FloatArray(side * side)
        // 左上 2×2 给高值 0.5，其余 0.1（整体没拉满，考验归一化）
        for (r in 0 until 2) for (c in 0 until 2) mask[r * side + c] = 0.5f
        for (i in mask.indices) if (mask[i] == 0f) mask[i] = 0.1f

        val grid = SaliencyMath.maskToGrid(mask, side, 2, 2)
        assertEquals(4, grid.size)
        assertEquals(1f, grid[0], 1e-5f)   // 左上格子全是被拉满的高值
        assertEquals(0f, grid[1], 1e-5f)   // 其余格子全是最低值
        assertEquals(0f, grid[2], 1e-5f)
        assertEquals(0f, grid[3], 1e-5f)
        assertEquals(1f, SaliencyMath.peak(grid), 1e-5f)
    }

    /** 整张掩膜一个值（模型什么都没看出来）→ 全 0，绝不能被"归一化"放大成一片主体 */
    @Test
    fun flatMaskYieldsEmptyGrid() {
        val side = 8
        val mask = FloatArray(side * side) { 0.37f }
        val grid = SaliencyMath.maskToGrid(mask, side, 4, 3)
        for (v in grid) assertEquals(0f, v, 1e-6f)
        assertEquals(0f, SaliencyMath.peak(grid), 1e-6f)
    }

    /**
     * 门槛守护测试：同样一张"背景全 0"的掩膜，
     *  - 主体是一大块（鼠标那种）→ 网格峰值接近 1，判"有主体"；
     *  - 只有几个像素的杂点（空场景那种）→ 被格子平均掉，峰值远低于门槛，判"没主体"。
     * 若有人把 [SaliencyMath.GRID_PEAK_GATE] 调没/调反，这条会红。
     */
    @Test
    fun peakGateSeparatesSubjectFromEmptyScene() {
        val side = 320
        val cols = 64
        val rows = 48

        // 有主体：右侧一块 96×140 的实心块（≈13% 面积），跟真机那张鼠标掩膜比例接近
        val withSubject = FloatArray(side * side)
        for (y in 60 until 200) for (x in 180 until 276) withSubject[y * side + x] = 1f
        val g1 = SaliencyMath.maskToGrid(withSubject, side, cols, rows)
        assertTrue("有主体时峰值应达到门槛：${SaliencyMath.peak(g1)}",
            SaliencyMath.peak(g1) >= SaliencyMath.GRID_PEAK_GATE)

        // 没主体：只有 3×3 个像素的杂点（渐晕边角那种）
        val empty = FloatArray(side * side)
        for (y in 100 until 103) for (x in 100 until 103) empty[y * side + x] = 1f
        val g2 = SaliencyMath.maskToGrid(empty, side, cols, rows)
        assertTrue("没有主体时峰值应低于门槛：${SaliencyMath.peak(g2)}",
            SaliencyMath.peak(g2) < SaliencyMath.GRID_PEAK_GATE)
    }

    /** 网格降到 AutoFrame 用的 64×48 后，行列数与坐标不能串（行优先） */
    @Test
    fun maskToGridIsRowMajor() {
        val side = 4
        val mask = FloatArray(side * side)
        // 只让右下角那个像素最高
        mask[3 * side + 3] = 1f
        val grid = SaliencyMath.maskToGrid(mask, side, 2, 2)
        // 降到 2×2 后右下格子是唯一非零
        assertEquals(0f, grid[0], 1e-6f)
        assertEquals(0f, grid[1], 1e-6f)
        assertEquals(0f, grid[2], 1e-6f)
        assertTrue(grid[3] > 0f)
    }

    /**
     * ★ 实时引导的坐标换算：显示方向 (u,v) → 传感器方向 (su,sv)。
     *
     * 拿四个角当"指纹"：只要映射写反（比如把 90° 写成 270°），四角就会整体对角互换，
     * 屏幕上"往左挪"的提示就会指到反方向 —— 这是最容易被忽略、又最影响手感的一处。
     */
    @Test
    fun displayToSensorMapsCorners() {
        val out = FloatArray(2)

        // 0°：原样
        SaliencyMath.displayToSensor(0, 0.25f, 0.75f, out)
        assertEquals(0.25f, out[0], 1e-6f)
        assertEquals(0.75f, out[1], 1e-6f)

        // 90°（顺时针转正）：su = v, sv = 1-u
        SaliencyMath.displayToSensor(90, 0f, 0f, out)
        assertEquals(0f, out[0], 1e-6f)
        assertEquals(1f, out[1], 1e-6f)
        SaliencyMath.displayToSensor(90, 1f, 1f, out)
        assertEquals(1f, out[0], 1e-6f)
        assertEquals(0f, out[1], 1e-6f)

        // 270°（逆时针转正）：su = 1-v, sv = u
        SaliencyMath.displayToSensor(270, 0f, 0f, out)
        assertEquals(1f, out[0], 1e-6f)
        assertEquals(0f, out[1], 1e-6f)

        // 180°：两个方向都翻
        SaliencyMath.displayToSensor(180, 0.2f, 0.3f, out)
        assertEquals(0.8f, out[0], 1e-6f)
        assertEquals(0.7f, out[1], 1e-6f)

        // 360° / -90° 归一化后等价于 0° / 270°
        SaliencyMath.displayToSensor(360, 0.25f, 0.75f, out)
        assertEquals(0.25f, out[0], 1e-6f)
        assertEquals(0.75f, out[1], 1e-6f)
        SaliencyMath.displayToSensor(-90, 0f, 0f, out)
        assertEquals(1f, out[0], 1e-6f)
        assertEquals(0f, out[1], 1e-6f)
    }

    /**
     * 与 `FaceAnalyzer.rotateGrid` 的一致性：显示方向右上角 (1,0) 在 90° 下回到传感器左上。
     * rotateGrid 里 90° 的公式是 dc = rows-1-r / dr = c，反解出来正是 su=v、sv=1-u。
     */
    @Test
    fun displayToSensorMatchesGridRotation() {
        val out = FloatArray(2)
        // 3 列 × 2 行的网格：rotateGrid 会把传感器 (0,0)（左上）搬到显示 (1,0)（右上）
        SaliencyMath.displayToSensor(90, 1f, 0f, out)
        assertEquals(0f, out[0], 1e-6f)
        assertEquals(0f, out[1], 1e-6f)
    }
}
