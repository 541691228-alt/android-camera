package cn.yege.dshcam

import android.annotation.SuppressLint
import android.media.Image
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 单帧观测结果
 * @param faceNorm 归一化到 [0,1] 的人脸框（已按旋转校正），无人脸时为 null
 * @param faceCount 检测到的人脸数量
 * @param meanLuma Y 平面平均亮度，归一化到 [0,1]
 * @param overexposedRatio Y>235 的像素占比
 * @param assist 峰值对焦/斑马纹用的亮度格子网格（已按旋转校正；未开启辅助时为 null）
 */
data class FrameObservation(
    val faceNorm: FaceBox?,
    val faceCount: Int,
    val meanLuma: Float?,
    val overexposedRatio: Float?,
    val assist: FocusAssist.AssistGrid? = null
)

/**
 * 基于 ML Kit 的人脸分析器，同时计算亮度统计。
 * 回调在 ML Kit 的分析线程触发，调用方需自行处理线程切换。
 * @param wantAssist 返回 true 时才计算「峰值对焦/斑马纹」的格子网格（关掉时不做任何多余计算）；
 *   同时决定要不要跑「主体模型锚点」（见 [anchor]）
 */
class FaceAnalyzer(
    private val onResult: (FrameObservation) -> Unit,
    private val wantAssist: () -> Boolean = { false }
) : androidx.camera.core.ImageAnalysis.Analyzer {

    companion object {
        private const val TAG = "CameraController"

        /** 主体模型锚点的刷新周期：1.5 秒一次（用户选的就是「模型当 1–2 秒锚点、帧间走规则」） */
        const val ANCHOR_PERIOD_MS = 1500L

        /** 锚点保鲜期：比刷新周期宽一点，掉一两帧也不至于立刻退回规则 */
        const val ANCHOR_TTL_MS = 3000L

        /** 锚点用的网格尺寸，与拍照链路保持一致 */
        private const val GW = 64
        private const val GH = 48
    }

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .build()
    )

    /** 跑模型的单线程执行器：推理几百毫秒，绝不能占着分析线程（那会掉帧） */
    private val modelExec = Executors.newSingleThreadExecutor { r -> Thread(r, "subject-anchor") }

    private val modelBusy = AtomicBoolean(false)

    @Volatile
    private var lastModelAt = 0L

    @Volatile
    private var lastVerdict: Boolean? = null

    /**
     * ★ 2026-10-05：逐锚点的细节日志开关。默认关（1.5 秒一行会把 logcat 刷满、也费电），
     * 由 MainActivity 用 `adb shell am start … --ez guidedbg true` 打开。
     */
    @Volatile
    var debug = false

    /**
     * 最近一次模型锚点（显示方向）。调用方用
     * `now - anchor.atMs < [ANCHOR_TTL_MS]` 判新鲜度，过期就回退到规则网格。
     */
    @Volatile
    var anchor: SubjectAnchor? = null
        private set

    /** 相机停止时调用：停掉模型线程，避免留着线程和会话 */
    fun release() {
        try {
            modelExec.shutdownNow()
        } catch (_: Throwable) {
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(image: ImageProxy) {
        val mediaImage = image.image
        if (mediaImage == null) {
            image.close()
            return
        }
        val rotationDegrees = image.imageInfo.rotationDegrees
        val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)

        // 先在当前帧上计算亮度统计（同步、轻量）
        val lumaPair = computeLumaStats(mediaImage)
        // ★ 峰值对焦/斑马纹的格子网格（只有辅助打开时才算）
        val grid = computeAssistGrid(mediaImage, rotationDegrees)
        // ★ 主体模型锚点（1.5 秒一次，异步；关了引导就完全不跑）
        maybeRunAnchor(mediaImage, rotationDegrees)

        val task = detector.process(inputImage)
            .addOnSuccessListener { faces ->
                val faceCount = faces.size
                val faceNorm = if (faces.isNotEmpty()) {
                    val box = faces[0].boundingBox
                    val isRotated = rotationDegrees == 90 || rotationDegrees == 270
                    val rotatedWidth = if (isRotated) image.height else image.width
                    val rotatedHeight = if (isRotated) image.width else image.height
                    val rw = rotatedWidth.toFloat()
                    val rh = rotatedHeight.toFloat()
                    if (rw > 0f && rh > 0f) {
                        FaceBox(
                            left = box.left.toFloat() / rw,
                            top = box.top.toFloat() / rh,
                            right = box.right.toFloat() / rw,
                            bottom = box.bottom.toFloat() / rh
                        )
                    } else null
                } else null
                onResult(FrameObservation(faceNorm, faceCount, lumaPair?.first, lumaPair?.second, grid))
            }
            .addOnFailureListener {
                onResult(FrameObservation(null, 0, lumaPair?.first, lumaPair?.second, grid))
            }
            .addOnCompleteListener { image.close() }
    }

    /**
     * 把传感器方向的 YUV 帧按**显示方向**降采样成 320×320 ARGB —— 主体模型的输入。
     *
     * 为什么不在分析线程里直接跑模型：手机上一次推理 500–850 ms，占着分析线程会掉帧；
     * 所以这里只做"取像素"（10 ms 量级），推理丢给 [modelExec]。
     * YUV_420_888 → RGB 用 BT.601 全范围公式（相机 YUV 是 full-range），
     * 色度取最近邻的 1/4 分辨率（源图 UV 本来就是 2×2 子采样，看不出来）。
     */
    private fun sampleArgb(mediaImage: Image, degrees: Int): IntArray? {
        val planes = mediaImage.planes
        if (planes.size < 3) return null
        val srcW = mediaImage.width
        val srcH = mediaImage.height
        if (srcW <= 0 || srcH <= 0) return null
        val yP = planes[0]
        val uP = planes[1]
        val vP = planes[2]
        val yb = yP.buffer
        val ub = uP.buffer
        val vb = vP.buffer
        val yRow = yP.rowStride
        val yPix = yP.pixelStride
        val uRow = uP.rowStride
        val uPix = uP.pixelStride
        val vRow = vP.rowStride
        val vPix = vP.pixelStride
        val yLimit = yb.limit()
        val uLimit = ub.limit()
        val vLimit = vb.limit()

        val side = SaliencyMath.SIDE
        val out = IntArray(side * side)
        val uv = FloatArray(2)
        for (oy in 0 until side) {
            val v0 = (oy + 0.5f) / side
            val rowBase = oy * side
            for (ox in 0 until side) {
                val u0 = (ox + 0.5f) / side
                SaliencyMath.displayToSensor(degrees, u0, v0, uv)
                val sx = (uv[0] * srcW).toInt().coerceIn(0, srcW - 1)
                val sy = (uv[1] * srcH).toInt().coerceIn(0, srcH - 1)
                val yi = sy * yRow + sx * yPix
                if (yi < 0 || yi >= yLimit) {
                    out[rowBase + ox] = 0xFF000000.toInt()
                    continue
                }
                val y = yb.get(yi).toInt() and 0xFF
                val ci = (sy / 2) * uRow + (sx / 2) * uPix
                val u = if (ci in 0 until uLimit) (ub.get(ci).toInt() and 0xFF) - 128 else 0
                val cj = (sy / 2) * vRow + (sx / 2) * vPix
                val vch = if (cj in 0 until vLimit) (vb.get(cj).toInt() and 0xFF) - 128 else 0
                val r = (y + 1.402f * vch).toInt().coerceIn(0, 255)
                val g = (y - 0.344136f * u - 0.714136f * vch).toInt().coerceIn(0, 255)
                val b = (y + 1.772f * u).toInt().coerceIn(0, 255)
                out[rowBase + ox] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return out
    }

    /** 节流 + 丢帧保护：1.5 秒一次，上一发还没跑完就不再排队 */
    private fun maybeRunAnchor(mediaImage: Image, degrees: Int) {
        if (!wantAssist() || !SubjectModel.isReady()) return
        if (modelBusy.get()) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastModelAt < ANCHOR_PERIOD_MS) return
        val argb = try {
            sampleArgb(mediaImage, degrees)
        } catch (t: Throwable) {
            Log.w(TAG, "主体锚点：取帧失败（${t.localizedMessage}）")
            null
        } ?: return
        lastModelAt = now
        modelBusy.set(true)
        try {
            modelExec.execute {
                try {
                    runAnchor(argb)
                } catch (t: Throwable) {
                    Log.w(TAG, "主体锚点：推理异常（${t.localizedMessage}）")
                } finally {
                    modelBusy.set(false)
                }
            }
        } catch (t: Throwable) {
            modelBusy.set(false)
        }
    }

    /**
     * 跑一次模型 → 主体质心 → [anchor]。
     *
     * 只在"有没有主体"这个判断翻转时打一行日志（1.5 秒一行会把 logcat 刷满），
     * 换句话说我盯着 logcat 就能看到它什么时候看懂了画面。
     */
    private fun runAnchor(argb: IntArray) {
        val t0 = SystemClock.elapsedRealtime()
        val grid = SubjectModel.salienceGridFromArgb(argb, SaliencyMath.SIDE, GW, GH) ?: return
        val subj = AutoFrame.subjectCenter(AutoFrame.Salience(GW, GH, grid))
        val peak = SubjectModel.lastPeak
        val has = peak >= SaliencyMath.GRID_PEAK_GATE
        val a = SubjectAnchor(
            x = subj?.x ?: 0.5f,
            y = subj?.y ?: 0.5f,
            strength = subj?.strength ?: 0f,
            spread = subj?.spread ?: 1f,
            peak = peak,
            atMs = t0,
            hasSubject = has
        )
        anchor = a
        val cost = SystemClock.elapsedRealtime() - t0
        // 逐锚点的细节日志只在 --ez guidedbg true 时开（1.5 秒一行会把 logcat 刷满）
        if (debug) {
            Log.d(
                TAG,
                "主体锚点[诊断]：%s 峰值=%.3f 主体=x=%.3f y=%.3f 强度=%.3f 扩散=%.3f 耗时=%dms".format(
                    if (has) "有主体" else "没有主体", peak, a.x, a.y, a.strength, a.spread, cost
                )
            )
        }
        if (lastVerdict == null || lastVerdict != has) {
            if (has) {
                Log.d(
                    TAG,
                    "主体锚点：有主体 峰值=%.3f 主体=x=%.3f y=%.3f 强度=%.3f 扩散=%.3f 耗时=%dms".format(
                        peak, a.x, a.y, a.strength, a.spread, cost
                    )
                )
            } else {
                Log.d(TAG, "主体锚点：没有主体 峰值=%.3f 耗时=%dms".format(peak, cost))
            }
        }
        lastVerdict = has
    }

    /**
     * 计算 Y 平面的平均亮度与过曝比例。
     * @return (meanLuma 0..1, overexposedRatio 0..1)，无法读取时返回 null
     */
    private fun computeLumaStats(mediaImage: Image): Pair<Float, Float>? {
        if (mediaImage.planes.isEmpty()) return null
        val plane = mediaImage.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val width = mediaImage.width
        val height = mediaImage.height
        if (width <= 0 || height <= 0) return null

        return try {
            val (mean, over) = computeLumaFromBuffer(buffer, width, height, rowStride, pixelStride)
            Pair(mean / 255f, over)
        } catch (e: Exception) {
            null
        }
    }

    private fun computeLumaFromBuffer(
        buffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int
    ): Pair<Float, Float> {
        var sum = 0L
        var overCount = 0L
        val total = width.toLong() * height.toLong()
        if (pixelStride == 1) {
            // 最常见情况：Y 平面紧凑排列，但行间可能有 padding
            val rowPadding = rowStride - width
            buffer.rewind()
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val y = buffer.get().toInt() and 0xFF
                    sum += y
                    if (y > 235) overCount++
                }
                if (rowPadding > 0) {
                    buffer.position(buffer.position() + rowPadding)
                }
            }
        } else {
            // 通用路径：按 pixelStride 跳跃读取
            buffer.rewind()
            for (row in 0 until height) {
                var rowOffset = row * rowStride
                for (col in 0 until width) {
                    val idx = rowOffset + col * pixelStride
                    if (idx + 1 <= buffer.limit()) {
                        val y = buffer.get(idx).toInt() and 0xFF
                        sum += y
                        if (y > 235) overCount++
                    }
                }
            }
        }
        val mean = if (total > 0) sum.toFloat() / total else 0f
        val overRatio = if (total > 0) overCount.toFloat() / total else 0f
        return Pair(mean, overRatio)
    }

    /**
     * 构建「峰值对焦 / 斑马纹」用的亮度格子网格。
     * ★ 关键：Y 平面是传感器方向的，而预览是转过 rotationDegrees 的 ——
     *   不把网格一起转过来，亮点/斜纹就会画在画面的另一边（转 90° 时宽高还要互换）。
     * 辅助关掉时直接返回 null，一帧都不多算。
     */
    private fun computeAssistGrid(mediaImage: Image, rotationDegrees: Int): FocusAssist.AssistGrid? {
        if (!wantAssist()) return null
        if (mediaImage.planes.isEmpty()) return null
        val plane = mediaImage.planes[0]
        val width = mediaImage.width
        val height = mediaImage.height
        if (width <= 0 || height <= 0) return null
        return try {
            val raw = FocusAssist.buildGrid(
                plane.buffer, width, height, plane.rowStride, plane.pixelStride
            )
            rotateGrid(raw, rotationDegrees)
        } catch (e: Exception) {
            null
        }
    }

    /** 把传感器方向的格子网格旋转到显示方向（0/90/180/270，其它角度按 0 处理） */
    private fun rotateGrid(
        g: FocusAssist.AssistGrid,
        degrees: Int
    ): FocusAssist.AssistGrid {
        val d = ((degrees % 360) + 360) % 360
        if (d == 0 || (d != 90 && d != 180 && d != 270)) return g
        val cols = g.cols
        val rows = g.rows
        val nc = if (d == 180) cols else rows
        val nr = if (d == 180) rows else cols
        val luma = IntArray(nc * nr)
        val grad = IntArray(nc * nr)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val src = r * cols + c
                val dc: Int
                val dr: Int
                when (d) {
                    90 -> { dc = rows - 1 - r; dr = c }        // 顺时针 90°：dst(x,y) = src(y, H-1-x)
                    180 -> { dc = cols - 1 - c; dr = rows - 1 - r }
                    else -> { dc = r; dr = cols - 1 - c }      // 270°（逆时针 90°）
                }
                val dst = dr * nc + dc
                luma[dst] = g.luma[src]
                grad[dst] = g.grad[src]
            }
        }
        return FocusAssist.AssistGrid(nc, nr, luma, grad)
    }
}
