package cn.yege.dshcam

/**
 * 自动曝光建议与防抖决策。
 *
 * 为什么这个类必须是纯 Kotlin（不 import 任何 android.* 类）：
 * 这里的逻辑只是几行加减法和阈值比较，和 Android 框架毫无关系。保持零 Android 依赖后，
 * 就能在 JVM 单元测试（src/test，普通 testImplementation + JUnit）里直接 new 出来跑，
 * 不需要模拟器、不需要 Robolectric、不需要 Android Gradle Plugin 的 androidTest 通道，
 * 单测毫秒级启动，阈值和防抖边界可以被穷举覆盖；反之如果这里碰了 Log、Context 之类的类，
 * 逻辑就只能上真机/模拟器验证，回归成本高、覆盖率低。
 *
 * 为什么"连续 5 帧同方向才动"：
 * 相机每帧的测光结果本身就在抖：画面里有人走过、自动白平衡与自动曝光互相干扰、
 * 增益调整的瞬间、甚至噪点都会让 meanLuma 在相邻帧之间跳几个百分点。如果一帧偏暗就立刻加 1/3 EV，
 * 下一帧又偏亮就立刻减回去，曝光会来回拉锯（俗称呼吸效应/oscillation），成片忽明忽暗，
 * 而且用户看到的预览亮度一直不稳定。所以引入有符号 streak 计数：必须连续 CONFIRM_FRAMES 帧
 * 都朝同一个方向建议，才真正动一次 STEP_EV；中途方向变了就重置为 ±1。这样只对持续存在的
 * 真实偏差做出响应，对单帧噪声不敏感。
 */
object AutoExposure {

    /** 曝光补偿的最大幅度（EV），超出后夹住不再动，避免把画面推到极端。 */
    const val LIMIT_EV = 1.0f

    /** 每次调整的步长（EV），1/3 EV 是相机业界常用的一档。 */
    const val STEP_EV = 1f / 3f

    /** 连续同方向多少帧才真正执行一次调整。 */
    const val CONFIRM_FRAMES = 5

    /** 平均亮度低于此值视为偏暗，需要加曝光。 */
    const val DARK_LUMA = 0.32f

    /** 过曝像素占比高于此值视为大片过曝，需要减曝光。 */
    const val BLOWN_HIGHLIGHT = 0.06f

    /** 一次决策的结果。 */
    data class Decision(val ev: Float, val streak: Int, val changed: Boolean)

    /**
     * 根据测光观测给出曝光补偿建议。
     *
     * @param meanLuma 全画面平均亮度，归一化到 [0,1]；无法计算时为 null
     * @param overexposedRatio 过曝像素占比，归一化到 [0,1]；无法计算时为 null
     * @return +STEP_EV 表示建议加曝光，-STEP_EV 表示建议减曝光，null 表示本轮不动
     */
    fun suggest(meanLuma: Float?, overexposedRatio: Float?): Float? {
        val dark = meanLuma != null && meanLuma < DARK_LUMA
        val blown = overexposedRatio != null && overexposedRatio > BLOWN_HIGHLIGHT

        // 既偏暗又有大片过曝：这是高动态范围场景（比如逆光的窗户 + 暗部人脸），
        // 单一曝光补偿无法同时救两头，硬调只会把一头推得更糟，所以这一轮不给建议。
        // 这种情况应该由 HDR 或用户手动点测光处理，不在本类的职责范围内。
        if (dark && blown) return null

        if (dark) return +STEP_EV
        if (blown) return -STEP_EV
        return null
    }

    /**
     * 曝光防抖决策：把逐帧的建议累积成有符号 streak，攒够 CONFIRM_FRAMES 帧才动一次。
     *
     * @param suggested suggest() 的输出，null 或 0f 表示本轮无方向
     * @param currentEv 当前已生效的曝光补偿（EV）
     * @param streak 上一轮的有符号计数：正数 = 连续想加曝光，负数 = 连续想减曝光，0 = 无方向
     */
    fun decide(suggested: Float?, currentEv: Float, streak: Int): Decision {
        // 没有建议（或建议为 0）就清空方向计数，重新攒。保持不变、不算改动。
        if (suggested == null || suggested == 0f) {
            return Decision(currentEv, 0, false)
        }

        val dir = if (suggested > 0f) 1f else -1f
        val dirInt = if (dir > 0f) 1 else -1

        // 与上一帧同方向 -> 计数继续累加（绝对值 +1）；方向反转 -> 从 ±1 重新开始。
        val newStreak = if (streak != 0 && (streak > 0) == (dirInt > 0)) {
            streak + dirInt
        } else {
            dirInt
        }

        // 还没攒够，先按住不动，只把计数带回去。
        if (newStreak > -CONFIRM_FRAMES && newStreak < CONFIRM_FRAMES) {
            return Decision(currentEv, newStreak, false)
        }

        // 攒够了，执行一次 STEP_EV，并夹到 ±LIMIT_EV。
        var ev = currentEv + dir * STEP_EV
        if (ev > LIMIT_EV) ev = LIMIT_EV
        if (ev < -LIMIT_EV) ev = -LIMIT_EV

        // 保留 3 位小数：STEP_EV 是 1/3 的循环小数，反复累加会积累浮点误差，
        // 导致比较 0.9999999 和 1.0 之类的边界出问题，也会让显示与实际不一致。
        ev = Math.round(ev * 1000f) / 1000f

        // 注意：即使被 LIMIT_EV 夹住（ev 没变），streak 也归 0。
        // 否则到了上限后 streak 会一直停在 5，每帧都走一遍这个分支，
        // 一旦用户手动降回范围内，会立刻又跳回上限，形成粘滞感。
        val changed = Math.abs(ev - currentEv) > 1e-4f
        return Decision(ev, 0, changed)
    }
}
