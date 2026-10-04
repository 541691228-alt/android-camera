package cn.yege.dshcam

import org.junit.Test
import org.junit.Assert.*

/**
 * AutoFrame 的 JUnit 4 单元测试。
 * 只依赖 Rules/AutoFrame 的纯 Kotlin 实现，不触碰任何 Android 类，可直接在 JVM 上跑。
 */
class AutoFrameTest {

    /** 造一张纯色 ARGB 图 */
    private fun solid(width: Int, height: Int, argb: Int): IntArray =
        IntArray(width * height) { argb }

    // 1) 均匀灰图：没有边缘、没有肤色、没有饱和度 → 显著性应全为 0，也就找不到主体
    @Test
    fun salienceFromArgbUniformGrayIsFlat() {
        val argb = solid(8, 8, 0xFF808080.toInt())
        val s = AutoFrame.salienceFromArgb(argb, 8, 8, 4, 4)
        assertEquals(4, s.cols)
        assertEquals(4, s.rows)
        for (r in 0 until 4) {
            for (c in 0 until 4) {
                assertEquals(0f, s.at(c, r), 1e-6f)
            }
        }
        assertNull(AutoFrame.subjectCenter(s))
    }

    // 2) 黑白棋盘：每格内部/格与格之间都强对比，edge 项很大 → 每格都该有明显权重
    //    ★ 2026-10-05 改成 1:1 网格（8×8 图 → 8×8 格）：生产里就是 64×48 图配 64×48 格，
    //    每格一个像素。原来用 4×4 格（每格 2×2 像素）会把棋盘平均成一片灰，
    //    中心-周边对比自然为 0，那是在测一个不存在的用法。
    @Test
    fun salienceFromArgbCheckerHasEdges() {
        val w = 8
        val h = 8
        val argb = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                argb[y * w + x] =
                    if ((x + y) % 2 == 0) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
            }
        }
        val s = AutoFrame.salienceFromArgb(argb, w, h, w, h)
        for (r in 0 until h) {
            for (c in 0 until w) {
                assertTrue("格($c,$r) 权重应大于 0.2，实际 ${s.at(c, r)}", s.at(c, r) > 0.2f)
            }
        }
    }

    // 3) 只在左上角放 2×2 肤色像素：该格命中 skin 项，应显著高于右下角的灰格
    @Test
    fun salienceFromArgbDetectsSkinPatch() {
        val w = 8
        val h = 8
        val argb = solid(w, h, 0xFF808080.toInt())
        for (y in 0..1) {
            for (x in 0..1) {
                argb[y * w + x] = 0xFFC08060.toInt()
            }
        }
        val s = AutoFrame.salienceFromArgb(argb, w, h, 8, 8)
        assertTrue("at(0,0) 应 >= 0.2，实际 ${s.at(0, 0)}", s.at(0, 0) >= 0.2f)
        assertTrue("肤色格应高于灰格", s.at(0, 0) > s.at(6, 6))
    }

    // 3b) 深色主体 + 带明暗渐变的纯色背景 —— 真机那张「黑色鼠标放在红色鼠标垫上」的抽象。
    //     守护 2026-10-05 的返工：只用「边缘/饱和度」或用「与全图主色的差」都判不出主体
    //     （渐晕背景会被当成主体、鼠标被淹没），换成中心-周边对比后才找得到。
    @Test
    fun salienceFromArgbFindsDarkSubjectOnVignettedBackground() {
        val w = 64
        val h = 48
        val argb = IntArray(w * h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                // 红垫：右上亮、左下暗（真机照片就是这种渐晕）
                val f = 0.55 + 0.45 * ((x.toDouble() / w) * 0.6 + (1.0 - y.toDouble() / h) * 0.4)
                val r = (0xC8 * f).toInt().coerceIn(0, 255)
                val g = (0x1E * f).toInt().coerceIn(0, 255)
                argb[y * w + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or g
            }
        }
        // 黑鼠标：右侧中间 12×12 一块
        for (y in 14 until 26) {
            for (x in 40 until 52) {
                argb[y * w + x] = 0xFF101010.toInt()
            }
        }
        val s = AutoFrame.salienceFromArgb(argb, w, h, w, h)
        val subject = AutoFrame.subjectCenter(s)
        assertNotNull("带渐晕的背景上应该能找出鼠标", subject)
        val sub = subject!!
        assertTrue("主体应落在右侧的鼠标上，实际 x=${sub.x}", sub.x > 0.55f)
        assertTrue("显著度应集中在鼠标一块，实际 spread=${sub.spread}", sub.spread < 0.25f)
        assertTrue("强度不能是 0，实际 ${sub.strength}", sub.strength > 0.02f)
    }

    // 3c) 纯色背景（连渐变都没有）：中心-周边对比处处为 0 → 不该凭空找出主体，
    //     否则自动构图会对着空墙乱裁。
    @Test
    fun salienceFromArgbFlatColoredBackgroundHasNoSubject() {
        val argb = solid(64, 48, 0xFFC81E1E.toInt())
        val s = AutoFrame.salienceFromArgb(argb, 64, 48, 64, 48)
        for (v in s.weight) assertEquals(0f, v, 1e-6f)
        assertNull(AutoFrame.subjectCenter(s))
    }

    // 3d) 竖拍（EXIF 6/8）：构图判定要在"用户看到的方向"上做，网格要能转正
    @Test
    fun rotateSalienceQuarterTurnSwapsAxes() {
        // 传感器 3 列 × 2 行，主体在传感器左上角 (0,0)
        val s = AutoFrame.Salience(3, 2, floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f))

        val cw = AutoFrame.rotateSalience(s, 6) // 顺时针 90° 转正 → 显示 2 列 × 3 行
        assertEquals(2, cw.cols)
        assertEquals(3, cw.rows)
        assertEquals("传感器左上 → 显示右上", 1f, cw.at(1, 0), 1e-6f)
        assertEquals(0f, cw.at(0, 0), 1e-6f)

        val ccw = AutoFrame.rotateSalience(s, 8) // 逆时针 90° → 传感器左上落到显示左下
        assertEquals(2, ccw.cols)
        assertEquals(3, ccw.rows)
        assertEquals("传感器左上 → 显示左下", 1f, ccw.at(0, 2), 1e-6f)

        val half = AutoFrame.rotateSalience(s, 3) // 180°：列行不变，点对角翻
        assertEquals(3, half.cols)
        assertEquals(2, half.rows)
        assertEquals(1f, half.at(2, 1), 1e-6f)

        val same = AutoFrame.rotateSalience(s, 1) // 正常方向：原样返回
        assertEquals(1f, same.at(0, 0), 1e-6f)
    }

    // 3e) 显示方向的裁剪框要能映射回传感器坐标（原图是按传感器方向解的、也只能那样裁）
    @Test
    fun mapCropToSensorInvertsQuarterTurn() {
        // 传感器 4000×3000（横）；竖拍成片显示为 3000×4000
        val shown = AutoFrame.CropRect(left = 100, top = 200, width = 1500, height = 2000, score = 0.9f)

        val s6 = AutoFrame.mapCropToSensor(shown, 4000, 3000, 6)
        assertEquals(200, s6.left)
        // 竖拍时显示宽度 = 传感器高度（3000），所以是 3000 - left - width
        assertEquals(3000 - 100 - 1500, s6.top)
        assertEquals(2000, s6.width) // 90° → 宽高互换
        assertEquals(1500, s6.height)
        assertTrue("$s6 应落在原图内", s6.left >= 0 && s6.top >= 0 &&
            s6.left + s6.width <= 4000 && s6.top + s6.height <= 3000)
        assertEquals(0.9f, s6.score, 1e-6f)

        val s8 = AutoFrame.mapCropToSensor(shown, 4000, 3000, 8)
        assertEquals(4000 - 200 - 2000, s8.left)
        assertEquals(100, s8.top)
        assertEquals(2000, s8.width)
        assertEquals(1500, s8.height)

        val s0 = AutoFrame.mapCropToSensor(shown, 4000, 3000, 1)
        assertEquals(100, s0.left)
        assertEquals(200, s0.top)
        assertEquals(1500, s0.width)
        assertEquals(2000, s0.height)
    }

    // 4) 单点权重为 1：其余为 0 → 质心应落在该格中心
    @Test
    fun subjectCenterFindsWeightedCentroid() {
        val cols = 4
        val rows = 3
        val w = FloatArray(cols * rows) { 0f }
        w[1 * cols + 2] = 1f // (col=2, row=1)
        val s = AutoFrame.Salience(cols, rows, w)
        val sub = AutoFrame.subjectCenter(s)
        assertNotNull(sub)
        val subject = sub!!
        assertEquals(0.625f, subject.x, 0.01f)
        assertEquals(0.5f, subject.y, 0.01f)
        assertTrue("strength 应大于 0", subject.strength > 0f)
    }

    // 5) 全 0 权重 → 没有主体
    @Test
    fun subjectCenterNullOnFlat() {
        val s = AutoFrame.Salience(4, 3, FloatArray(12) { 0f })
        assertNull(AutoFrame.subjectCenter(s))
    }

    // 6) 亮度网格：打平后所有格相同，再抬高某格梯度值 → 该格成为最大
    @Test
    fun salienceFromGridHighlightsGradientCell() {
        val cols = 4
        val rows = 3
        val luma = IntArray(cols * rows) { 100 }
        val grad = IntArray(cols * rows) { 0 }
        val flat = AutoFrame.salienceFromGrid(luma, grad, cols, rows)
        val ref = flat.at(0, 0)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                assertEquals(ref, flat.at(c, r), 1e-6f)
            }
        }

        grad[2 * cols + 3] = 220 // (col=3, row=2)
        val s = AutoFrame.salienceFromGrid(luma, grad, cols, rows)
        var maxWeight = 0f
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                if (s.at(c, r) > maxWeight) maxWeight = s.at(c, r)
            }
        }
        assertEquals(maxWeight, s.at(3, 2), 1e-6f)
    }

    // 7) 主体偏右：需要把主体往左挪回三分点，也就是镜头往右移
    @Test
    fun guideTellsCameraToMoveRightWhenSubjectIsTooFarRight() {
        val subject = AutoFrame.Subject(0.9f, 0.5f, 0.1f, 0.1f)
        val g = AutoFrame.guide(subject, true)
        assertFalse(g.ok)
        assertTrue(g.dx < 0f)
        assertTrue("文本应提示向右，实际 ${g.text}", g.text.contains("右"))
    }

    // 8) 主体正好落在三分交点 → 构图合格，且目标点就是该交点
    @Test
    fun guideOkWhenSubjectOnThirdsPoint() {
        val subject = AutoFrame.Subject(2f / 3f, 1f / 3f, 0.1f, 0.1f)
        val g = AutoFrame.guide(subject, true)
        assertTrue("应判定合格，实际 text=${g.text} dx=${g.dx} dy=${g.dy}", g.ok)
        assertTrue(g.text.contains("OK"))
        assertEquals(0.6667f, g.targetX, 0.01f)
        assertEquals(0.3333f, g.targetY, 0.01f)
    }

    // 9) 没有可信主体时不许再催人挪镜头（2026-10-04 用户反馈"永远往左一点往右一点"）：
    //    显著度铺满整幅（spread 大）→ confident=false、不给方向、也不画箭头
    @Test
    fun guideStaysQuietWhenNoClearSubject() {
        val diffuse = AutoFrame.Subject(0.42f, 0.5f, 0.235f, 0.549f)
        val g = AutoFrame.guide(diffuse, true)
        assertFalse("没有可信主体时不应说构图 OK", g.ok)
        assertFalse("没有可信主体时 confident 必须为 false", g.confident)
        assertFalse("没有可信主体时不应提示左右", g.text.contains("移一点"))
        assertFalse("没有可信主体时不应提示抬高/压低", g.text.contains("抬高") || g.text.contains("压低"))
        assertTrue("应给出中性提示，实际 ${g.text}", g.text.contains("主体"))

        // 强度太弱（几乎没有亮点）同样不给方向
        val weak = AutoFrame.Subject(0.8f, 0.2f, 0.02f, 0.05f)
        assertFalse(AutoFrame.guide(weak, true).confident)
    }

    // 10) 有主体但没到位：要说清"还差多少"（用户抱怨的"没一个度"）
    @Test
    fun guideReportsHowFarToMove() {
        val subject = AutoFrame.Subject(0.60f, 1f / 3f, 0.30f, 0.10f)
        val g = AutoFrame.guide(subject, true)
        assertTrue(g.confident)
        assertFalse(g.framed)
        assertFalse(g.ok)
        assertTrue("应提示往左移，实际 ${g.text}", g.text.contains("往左移"))
        assertTrue("要带上距离百分比，实际 ${g.text}", g.text.contains("%"))
    }

    // 11) 迟滞：已经进了 6% 容差就判达标；偏到 8% 时，只有"上一帧已达标"才继续算达标
    @Test
    fun guideUsesHysteresisAroundTolerance() {
        val justInside = AutoFrame.Subject(2f / 3f - 0.04f, 1f / 3f, 0.30f, 0.10f)
        val a = AutoFrame.guide(justInside, true)
        assertTrue("偏 4% 应算构图达标", a.framed)
        assertTrue(a.ok)

        val slightlyOut = AutoFrame.Subject(2f / 3f - 0.08f, 1f / 3f, 0.30f, 0.10f)
        val b = AutoFrame.guide(slightlyOut, true, false)
        assertFalse("偏 8% 且上一帧没达标 → 应继续提示", b.framed)
        val c = AutoFrame.guide(slightlyOut, true, true)
        assertTrue("偏 8% 但上一帧已达标 → 不该反复催（迟滞）", c.framed)
    }

    // 11b) 强度门限的迟滞：真机日志里规则的强度在 0.039 / 0.069 之间抖，
    //      只用一个 0.06 的门限会让文案在"给方向"和"没找到主体"之间一秒一跳。
    @Test
    fun guideKeepsTrustedSubjectWhenStrengthDips() {
        // 骑在门限上：0.05 —— 没被信任过 → 不算主体
        val borderline = AutoFrame.Subject(0.60f, 1f / 3f, 0.05f, 0.10f)
        val cold = AutoFrame.guide(borderline, true, false, false)
        assertFalse("还没信任过就该按 0.06 判：不算主体", cold.confident)
        assertTrue("应给中性提示，实际 ${cold.text}", cold.text.contains("主体"))

        // 同一个强度，但上一帧已经把它当可信主体 → 门限放宽到 0.045，继续给方向（不再横跳）
        val warm = AutoFrame.guide(borderline, true, false, true)
        assertTrue("已经信任过就不该因为掉到 0.045–0.06 之间而改口", warm.confident)
        assertTrue("应给方向，实际 ${warm.text}", warm.text.contains("移一点") || warm.text.contains("靠近"))

        // 掉到 0.045 以下就该收回（真的没主体了，不能让方向一直挂着）
        val gone = AutoFrame.Subject(0.60f, 1f / 3f, 0.03f, 0.10f)
        assertFalse(AutoFrame.guide(gone, true, false, true).confident)
    }

    // 12) 平整明亮的大背景（例如一整块纯色桌面）不该被当成主体：
    //     亮度项减掉均值后权重应≈0，直接判"没找到主体"，而不是永远催人挪镜头
    @Test
    fun salienceFromGridFlatBrightSceneHasNoSubject() {
        val cols = 8
        val rows = 6
        val s = AutoFrame.salienceFromGrid(
            IntArray(cols * rows) { 180 }, IntArray(cols * rows) { 0 }, cols, rows
        )
        var maxW = 0f
        for (w in s.weight) if (w > maxW) maxW = w
        assertTrue("平整画面的权重应≈0，实际 $maxW", maxW <= 0.001f)
        val sub = AutoFrame.subjectCenter(s)
        assertTrue("平整画面不该判出可信主体", sub == null || !AutoFrame.guide(sub, true).confident)
    }

    // 13) 高权重集中在最右两列 → 最佳裁剪框必须把这片区域完整包住，同时保持 1:1 比例
    @Test
    fun bestCropKeepsSubjectInsideAndHonorsAspect() {
        val cols = 8
        val rows = 6
        val w = FloatArray(cols * rows) { 0f }
        for (r in 0 until rows) {
            w[r * cols + 6] = 1f
            w[r * cols + 7] = 1f
        }
        val s = AutoFrame.Salience(cols, rows, w)
        val crop = AutoFrame.bestCrop(s, 1200, 900, 1f)

        assertTrue("框应覆盖到 x=900，实际 ${crop.left + crop.width}", crop.left + crop.width >= 900)
        assertEquals(1f, crop.width.toFloat() / crop.height.toFloat(), 0.02f)
        assertTrue(crop.left >= 0)
        assertTrue(crop.top >= 0)
        assertTrue(crop.left + crop.width <= 1200)
        assertTrue(crop.top + crop.height <= 900)
    }

    // 10) 全 0 权重是非法输入的一种兜底情形：返回居中最大框，绝不抛异常
    @Test
    fun bestCropFullZeroStaysCentered() {
        val s = AutoFrame.Salience(6, 4, FloatArray(24) { 0f })
        val crop = AutoFrame.bestCrop(s, 1600, 1200, 16f / 9f)

        assertEquals(0f, crop.score, 1e-6f)
        assertTrue(Math.abs(crop.left - (1600 - crop.width) / 2) <= 1)
        assertTrue(Math.abs(crop.top - (1200 - crop.height) / 2) <= 1)
        assertEquals(16f / 9f, crop.width.toFloat() / crop.height.toFloat(), 0.02f)
    }

    // 11) ★ 2026-10-05 人工补的回归测试：主体在画面中间偏一点时，
    //     bestCrop 必须真的裁一刀（把主体挪到三分点），而不是永远返回整图。
    //     背景：修 edgePenalty 的归一化 + 覆盖/三分/面积三档权重配平之前，
    //     真机后置/前置两张照片都判"构图已达标、不另存"，自动构图等于没生效。
    @Test
    fun bestCropRecomposesOffCenterSubject() {
        val cols = 64
        val rows = 48
        val w = FloatArray(cols * rows) { 0.02f }
        // 主体：中间偏左上一小片高显著度（12×12 格）
        for (r in 18 until 30) {
            for (c in 26 until 38) {
                w[r * cols + c] = 1f
            }
        }
        val s = AutoFrame.Salience(cols, rows, w)
        val crop = AutoFrame.bestCrop(s, 1200, 900, 4f / 3f, 0.60f)

        val area = crop.width.toDouble() * crop.height.toDouble() / (1200.0 * 900.0)
        assertTrue("应当真的裁掉一部分，而不是原样返回（实际保留面积 %.3f)".format(area), area < 0.99)
        assertTrue("裁剪框必须仍然含住主体（left=${crop.left} width=${crop.width}）",
            crop.left <= 26 * 1200 / cols && crop.left + crop.width >= 38 * 1200 / cols)
        assertTrue("裁剪框必须落在图内", crop.left >= 0 && crop.top >= 0 &&
            crop.left + crop.width <= 1200 && crop.top + crop.height <= 900)
        assertEquals("宽高比要保持 4:3", 4f / 3f, crop.width.toFloat() / crop.height.toFloat(), 0.03f)
    }

    /**
     * ★ 2026-10-05 新增：主体本来就在三分点上时，别为了零点几分硬裁一刀。
     * 这条同时守住"动剪刀门槛"（bestCrop 里 bestScore >= baseScore + 0.05 才采纳裁剪），
     * 门槛一旦被改没，这条测试会立刻红。
     */
    @Test
    fun bestCropKeepsFrameWhenSubjectAlreadyOnThirds() {
        val cols = 64
        val rows = 48
        val w = FloatArray(cols * rows) { 0.02f }
        // 主体 12×12，中心正好压在左上三分交点（约 1/3, 1/3）上
        for (r in 10 until 22) {
            for (c in 15 until 27) {
                w[r * cols + c] = 1f
            }
        }
        val s = AutoFrame.Salience(cols, rows, w)
        val crop = AutoFrame.bestCrop(s, 1200, 900, 4f / 3f, 0.60f)

        val area = crop.width.toDouble() * crop.height.toDouble() / (1200.0 * 900.0)
        assertTrue("主体已在三分点，不该再裁（实际保留面积 %.3f，判定=%s)".format(area, AutoFrame.lastScoreInfo),
            area >= 0.97)
        assertTrue("返回的框仍要含住主体", crop.left <= 15 * 1200 / cols &&
            crop.left + crop.width >= 27 * 1200 / cols)
        assertTrue("应当是「保持最大框」而不是「采纳裁剪」（实际 %s）".format(AutoFrame.lastScoreInfo),
            AutoFrame.lastScoreInfo.startsWith("保持最大框"))
    }
}
