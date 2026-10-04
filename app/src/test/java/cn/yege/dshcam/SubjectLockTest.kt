package cn.yege.dshcam

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ★ 2026-10-05：守护「模型锚点不横跳」这条需求 —— 用户原话是
 * 「ai构图提示吃没一个度吗 永远是向右一点向左一点靠近一点没一个度吗」。
 *
 * 真机日志（19:30）证明主体模型给的位置会每 1.5 秒换一个对象，直接拿来当主体就会让提示横跳；
 * [SubjectLock] 就是那层迟滞。这个测试文件红了＝提示又开始横跳了。
 */
class SubjectLockTest {

    private fun anchor(
        x: Float,
        y: Float,
        strength: Float = 0.20f,
        has: Boolean = true,
        atMs: Long = 0L,
    ) = SubjectAnchor(x, y, strength, 0.10f, 0.90f, atMs, has)

    @Test
    fun locksOnlyAfterTwoAgreeingAnchors() {
        val lock = SubjectLock()
        // 第一个锚点：不敢信（它可能是抓错了对象）
        assertFalse(lock.update(anchor(0.30f, 0.30f)))
        assertFalse(lock.locked)
        // 第二个锚点跟第一个对得上：锁定在两点的中点上
        assertTrue(lock.update(anchor(0.32f, 0.31f)))
        assertEquals(0.31f, lock.x, 1e-4f)
        assertEquals(0.305f, lock.y, 1e-4f)
    }

    @Test
    fun singleJitterDoesNotMoveTheLock() {
        val lock = SubjectLock()
        lock.update(anchor(0.30f, 0.30f))
        lock.update(anchor(0.30f, 0.30f))
        assertTrue(lock.locked)

        // 模型突然跳到画面另一头（换了对象）：一次不算数，位置不许动
        assertTrue(lock.update(anchor(0.80f, 0.85f)))
        assertEquals(0.30f, lock.x, 1e-4f)
        assertEquals(0.30f, lock.y, 1e-4f)

        // 第二次还是那头：这才认账，改锁
        assertTrue(lock.update(anchor(0.80f, 0.85f)))
        assertEquals(0.80f, lock.x, 1e-4f)
        assertEquals(0.85f, lock.y, 1e-4f)

        // 改锁之后又对得上，就稳在那儿
        assertTrue(lock.update(anchor(0.80f, 0.85f)))
        assertEquals(0.80f, lock.x, 1e-4f)
    }

    @Test
    fun unlocksAfterTwoEmptyAnchors() {
        val lock = SubjectLock()
        lock.update(anchor(0.30f, 0.30f))
        lock.update(anchor(0.30f, 0.30f))
        assertTrue(lock.locked)

        // 单次"没有主体"不解锁（模型偶尔会抖）
        assertTrue(lock.update(anchor(0f, 0f, has = false)))
        // 连续两次：解锁 → 调用方回退规则网格
        assertFalse(lock.update(anchor(0f, 0f, has = false)))
        assertFalse(lock.locked)
        assertTrue(lock.x < 0f)
    }

    @Test
    fun weakStrengthCountsAsNoSubject() {
        val lock = SubjectLock()
        // 强度 0.02＝掩膜太散（真机纯红垫就是这种），即使模型说"有主体"也不算
        assertFalse(lock.update(anchor(0.30f, 0.30f, strength = 0.02f)))
        assertFalse(lock.update(anchor(0.31f, 0.30f, strength = 0.02f)))
        assertFalse(lock.locked)

        // 弱锚点不该污染候选：随后两个正常锚点照常锁定
        assertFalse(lock.update(anchor(0.30f, 0.30f, strength = 0.20f)))
        assertTrue(lock.update(anchor(0.31f, 0.30f, strength = 0.20f)))
        assertTrue(lock.locked)
    }

    @Test
    fun neverLocksWhenAnchorsAlternateBetweenTwoSpots() {
        val lock = SubjectLock()
        // 杂乱场景：锚点在两个对象之间来回跳 → 永远对不上，永远不锁定
        // （这时引导该走规则网格，而不是跟着模型乱指）
        repeat(6) {
            assertFalse(lock.update(anchor(0.20f, 0.20f)))
            assertFalse(lock.update(anchor(0.80f, 0.80f)))
        }
        assertFalse(lock.locked)
    }

    @Test
    fun clearResetsAndCanLockAgain() {
        val lock = SubjectLock()
        lock.update(anchor(0.30f, 0.30f))
        lock.update(anchor(0.30f, 0.30f))
        assertTrue(lock.locked)

        lock.clear()
        assertFalse(lock.locked)

        // 清干净之后要重新走"两个锚点对得上"
        assertFalse(lock.update(anchor(0.60f, 0.60f)))
        assertTrue(lock.update(anchor(0.61f, 0.60f)))
        assertEquals(0.605f, lock.x, 1e-4f)
    }
}
