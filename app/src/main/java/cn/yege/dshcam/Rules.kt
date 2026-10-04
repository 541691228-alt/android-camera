package cn.yege.dshcam

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import java.util.Locale

data class FaceBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f
}

data class FrameFacts(
    val viewW: Int,
    val viewH: Int,
    val face: FaceBox?,
    val faceCount: Int,
    val pitchDeg: Float,
    val rollDeg: Float,
    val horizonY: Float?,
    val meanLuma: Float?,
    val overexposedRatio: Float
)

data class Advice(
    val score: Int,
    val lines: List<String>,
    val levelOk: Boolean,
    val thirdsOffsetX: Float
)

object Rules {
    fun evaluate(f: FrameFacts): Advice {
        var score = 70
        val msgs = mutableListOf<String>()
        val rollAbs = abs(f.rollDeg)
        val levelOk = rollAbs <= 1f
        
        var thirdsOffsetX = 0f

        // 1. 水平检测 (最高优先级)
        if (!levelOk) {
            score -= 20
            val msg = "水平差 ${String.format(Locale.US, "%.1f", rollAbs)}°，${if (rollAbs > 3f) "端平手机" else "稍微端平"}"
            msgs.add(msg)
        }

        // 2. 曝光检测 (高优先级)
        var expAdded = false
        if (f.meanLuma != null && f.meanLuma < 0.32f) {
            score -= 12
            msgs.add("画面偏暗，+0.3EV")
            expAdded = true
        }
        if (f.overexposedRatio > 0.06f) {
            if (!expAdded) {
                score -= 12
                msgs.add("高光过曝，-0.3EV")
            }
        }

        // 3. 主体相关检测
        if (f.face != null) {
            val fb = f.face
            
            // 大小检测
            if (fb.height < 0.12f) {
                score -= 10
                msgs.add("主体偏小，靠近两步或用长焦")
            } else if (fb.height > 0.70f) {
                score -= 10
                msgs.add("太近了，退半步")
            }

            // 三分法检测
            val cx = fb.centerX
            val d1 = abs(cx - 0.333f)
            val d2 = abs(cx - 0.666f)
            thirdsOffsetX = min(d1, d2)
            if (thirdsOffsetX > 0.06f) {
                score -= 10
                val dir = if (cx < 0.5f) "向右" else "向左"
                msgs.add("主体${dir}移一点，落到三分线上")
            }

            // 地平线检测
            if (f.horizonY != null) {
                val hy = f.horizonY
                val h1 = abs(hy - 0.333f)
                val h2 = abs(hy - 0.666f)
                val ideal = if (h1 < h2) 0.333f else 0.666f
                if (abs(hy - ideal) > 0.08f) {
                    score -= 8
                    val dir = if (hy < ideal) "压低" else "抬高"
                    msgs.add("地平线${dir}")
                }
            }
        } else {
            // 无主体情况
            msgs.add("未识别到主体")
        }

        // 完美分数修正
        if (msgs.isEmpty() && levelOk) {
            score = 88
        }

        // 限制分数范围
        score = max(0, min(100, score))

        // 取前两条建议
        return Advice(score, msgs.take(2), levelOk, thirdsOffsetX)
    }
}
