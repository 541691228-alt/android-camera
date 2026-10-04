package cn.yege.dshcam

import kotlin.math.abs

/**
 * ★ 2026-10-05：模型锚点的「锁定 + 迟滞」状态机（纯 Kotlin，无 android import，可 JVM 单测）。
 *
 * 为什么需要它：真机 19:30 的日志里，同一个杂乱桌面场景，u2netp 每 1.5 秒会在不同的东西上跳
 * （玩具球 → 显示器 → RGB 风扇），锚点质心 x 在 0.276–0.541 之间跳、y 在 0.386–0.905 之间跳，
 * 于是构图提示几秒内横跳一遍：
 * `构图 OK` → `镜头压低一点（约 16%）` → `镜头往右移一点（约 9%）` → `没找到明显主体` → `构图 OK`
 * —— 正是用户抱怨的"永远是向右一点向左一点，没一个度"。
 *
 * 规则（三条，缺一不可）：
 *  1. **不急着信单个锚点**：连续两个锚点位置互差 ≤ [tol] 才认账锁定（单个锚点可能是它抓错了对象）。
 *  2. **锁住就别乱动**：锁定后位置做半步融合；连续 [missLimit] 个锚点都对不上才改锁（真的换了拍摄对象才跟）。
 *  3. **说没有就是没有**：`hasSubject=false` 或强度 < [minStrength]（掩膜太散）算"没有主体"，
 *     连续 [missLimit] 次就解锁 —— 交给规则网格（它的稳定性此前已在真机验证过）。
 *
 * 锁定失败时调用方应当回退到规则网格，而不是拿这个位置硬给方向。
 */
class SubjectLock(
    private val tol: Float = 0.08f,
    private val minStrength: Float = 0.06f,
    private val missLimit: Int = 2,
) {
    /** 锁定位置（归一化显示坐标）；< 0 表示当前没有锁定主体 */
    var x: Float = -1f
        private set
    var y: Float = -1f
        private set

    val locked: Boolean get() = x >= 0f

    /** 还在"连续两个锚点对得上"的观察期里的候选位置 */
    private var candX = -1f
    private var candY = -1f

    /** 连续多少个锚点与锁定位置对不上（或连续多少次说没有主体） */
    private var miss = 0

    /**
     * 喂一个新锚点，返回 true＝现在有一个可信的锁定主体（[x]/[y] 可用）。
     */
    fun update(a: SubjectAnchor): Boolean {
        val confident = a.hasSubject && a.strength >= minStrength
        if (confident) {
            if (!locked) {
                if (candX >= 0f && abs(a.x - candX) <= tol && abs(a.y - candY) <= tol) {
                    x = (a.x + candX) / 2f
                    y = (a.y + candY) / 2f
                    miss = 0
                } else {
                    candX = a.x
                    candY = a.y
                }
            } else if (abs(a.x - x) <= tol && abs(a.y - y) <= tol) {
                x = x * 0.5f + a.x * 0.5f
                y = y * 0.5f + a.y * 0.5f
                miss = 0
            } else {
                miss++
                if (miss >= missLimit) {
                    x = a.x
                    y = a.y
                    miss = 0
                    candX = -1f
                    candY = -1f
                }
            }
        } else if (locked) {
            miss++
            if (miss >= missLimit) clear()
        }
        return locked
    }

    fun clear() {
        x = -1f
        y = -1f
        candX = -1f
        candY = -1f
        miss = 0
    }
}
