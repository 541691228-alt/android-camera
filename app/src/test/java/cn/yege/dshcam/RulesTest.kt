package cn.yege.dshcam

import org.junit.Test
import org.junit.Assert.*

/**
 * Rules.kt 的单元测试
 * 运行环境：JVM，无需 Android 依赖
 */
class RulesTest {

    /**
     * 测试水平合格情况
     * 条件：rollDeg = 0.5f (接近水平)
     * 期望：levelOk = true
     */
    @Test
    fun testLevelOk() {
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = null,
            faceCount = 0,
            pitchDeg = 0f,
            rollDeg = 0.5f,
            horizonY = null,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertTrue("水平角度 0.5 度应判定为合格", advice.levelOk)
    }

    /**
     * 测试右倾 5 度情况
     * 条件：rollDeg = 5.0f
     * 期望：levelOk = false 且 lines 非空
     */
    @Test
    fun testTiltRight() {
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = null,
            faceCount = 0,
            pitchDeg = 0f,
            rollDeg = 5.0f,
            horizonY = null,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertFalse("倾斜 5 度应判定为不合格", advice.levelOk)
        assertTrue("不合格时应提供建议文字", advice.lines.isNotEmpty())
    }

    /**
     * 测试主体居中偏右情况
     * 条件：face 中心 x = 0.5 (正中间)
     * 期望：thirdsOffsetX > 0.06 (偏离三分线)
     */
    @Test
    fun testSubjectCenterRight() {
        val face = FaceBox(left = 0.4f, top = 0.4f, right = 0.6f, bottom = 0.6f)
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = face,
            faceCount = 1,
            pitchDeg = 0f,
            rollDeg = 0f,
            horizonY = null,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertTrue("居中主体应检测到偏离三分线的偏移量", advice.thirdsOffsetX > 0.06f)
    }

    /**
     * 测试主体偏小情况
     * 条件：face 高度 = 0.05
     * 期望：建议中包含关于大小的提示
     */
    @Test
    fun testSubjectSmall() {
        val face = FaceBox(left = 0.45f, top = 0.45f, right = 0.55f, bottom = 0.50f)
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = face,
            faceCount = 1,
            pitchDeg = 0f,
            rollDeg = 0f,
            horizonY = null,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertTrue("主体过小时应有相关提示", advice.lines.any { it.contains("小") })
    }

    /**
     * 测试人脸太大情况
     * 条件：face 高度 = 0.8
     * 期望：建议中包含关于大小的提示
     */
    @Test
    fun testFaceBig() {
        val face = FaceBox(left = 0.1f, top = 0.1f, right = 0.9f, bottom = 0.9f)
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = face,
            faceCount = 1,
            pitchDeg = 0f,
            rollDeg = 0f,
            horizonY = null,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        // Rules 对这一档给的文案是「太近了，退半步」（不是"大"字），断言别写死字面
        assertTrue("人脸过大时应有相关提示", advice.lines.any { it.contains("近") || it.contains("退") })
    }

    /**
     * 测试地平线过高情况
     * 条件：horizonY = 0.15
     * 期望：建议中包含关于地平线的提示
     */
    @Test
    fun testHorizonHigh() {
        // ★ 地平线检测在 Rules 里排在"主体大小/三分法"之后，而且 Advice 只取前 2 条，
        //   所以这里的主体必须同时满足：高度 0.12~0.70（不触发大小提示）
        //   且 centerX 落在 0.333 附近 ±0.06（不触发三分提示），否则地平线提示会被挤掉。
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = FaceBox(left = 0.30f, top = 0.40f, right = 0.37f, bottom = 0.60f),
            faceCount = 1,
            pitchDeg = 0f,
            rollDeg = 0f,
            horizonY = 0.15f,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertTrue("地平线过高时应有相关提示", advice.lines.any { it.contains("地平线") })
    }

    /**
     * 测试画面偏暗情况
     * 条件：meanLuma = 0.2
     * 期望：建议中包含关于亮度的提示
     */
    @Test
    fun testDark() {
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = null,
            faceCount = 0,
            pitchDeg = 0f,
            rollDeg = 0f,
            horizonY = null,
            meanLuma = 0.2f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertTrue("画面偏暗时应有相关提示", advice.lines.any { it.contains("暗") })
    }

    /**
     * 测试无主体情况
     * 条件：face = null
     * 期望：分数仍在 0 到 100 之间
     */
    @Test
    fun testNoSubject() {
        val facts = FrameFacts(
            viewW = 1080,
            viewH = 1920,
            face = null,
            faceCount = 0,
            pitchDeg = 0f,
            rollDeg = 0f,
            horizonY = null,
            meanLuma = 0.5f,
            overexposedRatio = 0f
        )
        val advice = Rules.evaluate(facts)
        assertTrue("无主体时分数应在有效范围内", advice.score in 0..100)
    }
}
