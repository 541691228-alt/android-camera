package cn.yege.dshcam

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * AutoExposure 的单元测试。
 * 全部为纯 Kotlin 逻辑测试，不依赖任何 android.* 类，可直接在 JVM 上运行。
 */
class AutoExposureTest {

    /** 意图：偏暗画面（meanLuma 低于 DARK_LUMA 且无过曝）应建议增加一档曝光 +STEP_EV */
    @Test
    fun suggestDarkSceneSuggestsPlusStep() {
        val ev = AutoExposure.suggest(0.2f, 0f)
        assertEquals(AutoExposure.STEP_EV, ev!!, 0.002f)
    }

    /** 意图：过曝画面（overexposedRatio 超过 BLOWN_HIGHLIGHT）应建议减少一档曝光 -STEP_EV */
    @Test
    fun suggestBlownHighlightsSuggestsMinusStep() {
        val ev = AutoExposure.suggest(0.5f, 0.2f)
        assertEquals(-AutoExposure.STEP_EV, ev!!, 0.002f)
    }

    /** 意图：亮度正常且无过曝时不应给出任何曝光调整建议，返回 null */
    @Test
    fun suggestNormalSceneReturnsNull() {
        assertNull(AutoExposure.suggest(0.5f, 0f))
    }

    /** 意图：既偏暗又过曝属于自相矛盾的画面，规则不做任何调整，返回 null */
    @Test
    fun suggestDarkAndBlownReturnsNull() {
        assertNull(AutoExposure.suggest(0.2f, 0.2f))
    }

    /** 意图：亮度与过曝数据都拿不到时无法判断，返回 null */
    @Test
    fun suggestNullInputsReturnsNull() {
        assertNull(AutoExposure.suggest(null, null))
    }

    /** 意图：同一方向必须连续 CONFIRM_FRAMES 帧才真正生效，前 4 帧只累计 streak 不改变曝光 */
    @Test
    fun decideRequiresConsecutiveFramesBeforeChanging() {
        val first = AutoExposure.decide(AutoExposure.STEP_EV, 0f, 0)
        assertFalse(first.changed)
        assertEquals(0f, first.ev, 0.002f)
        assertEquals(1, first.streak)

        val second = AutoExposure.decide(AutoExposure.STEP_EV, 0f, first.streak)
        assertFalse(second.changed)
        assertEquals(0f, second.ev, 0.002f)
        assertEquals(2, second.streak)

        val third = AutoExposure.decide(AutoExposure.STEP_EV, 0f, second.streak)
        assertFalse(third.changed)
        assertEquals(0f, third.ev, 0.002f)
        assertEquals(3, third.streak)

        val fourth = AutoExposure.decide(AutoExposure.STEP_EV, 0f, third.streak)
        assertFalse(fourth.changed)
        assertEquals(0f, fourth.ev, 0.002f)
        assertEquals(4, fourth.streak)

        val fifth = AutoExposure.decide(AutoExposure.STEP_EV, 0f, fourth.streak)
        assertTrue(fifth.changed)
        assertEquals(AutoExposure.STEP_EV, fifth.ev, 0.002f)
        assertEquals(0, fifth.streak)
    }

    /** 意图：调整方向中途翻转时旧的累计作废，streak 反向计数为 -1 且本帧不改曝光 */
    @Test
    fun decideDirectionFlipResetsStreak() {
        val up = AutoExposure.decide(AutoExposure.STEP_EV, 0f, 0)
        assertEquals(1, up.streak)

        val flipped = AutoExposure.decide(-AutoExposure.STEP_EV, 0f, up.streak)
        assertEquals(-1, flipped.streak)
        assertFalse(flipped.changed)
        assertEquals(0f, flipped.ev, 0.002f)
    }

    /** 意图：已到达 ±LIMIT_EV 边界且 streak 足够时仍不应越界，夹紧在边界值并清零 streak */
    @Test
    fun decideClampsAtEvLimits() {
        val upper = AutoExposure.decide(AutoExposure.STEP_EV, AutoExposure.LIMIT_EV, 4)
        assertEquals(AutoExposure.LIMIT_EV, upper.ev, 0.002f)
        assertFalse(upper.changed)
        assertEquals(0, upper.streak)

        val lower = AutoExposure.decide(-AutoExposure.STEP_EV, -AutoExposure.LIMIT_EV, -4)
        assertEquals(-AutoExposure.LIMIT_EV, lower.ev, 0.002f)
        assertFalse(lower.changed)
        assertEquals(0, lower.streak)
    }
}
