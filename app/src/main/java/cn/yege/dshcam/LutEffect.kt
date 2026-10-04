package cn.yege.dshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraEffect
import androidx.camera.core.SurfaceOutput
import androidx.camera.core.SurfaceProcessor
import androidx.camera.core.SurfaceRequest
import androidx.core.util.Consumer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executor

/** LUT PNG 的固定尺寸：1024x32（32 个 slice，每 slice 32x32）。GL 与 CPU 两条路共用。 */
private const val LUT_TEX_WIDTH = 1024
private const val LUT_TEX_HEIGHT = 32

/**
 * LUT 的格点数（32 个 slice，每 slice 32x32）。GL 与 CPU 两条路共用。
 *
 * 注意 [LUT_TEX_WIDTH] = `LUT_SLICES * LUT_SLICES` = 1024，别把这两个 32 弄混。
 */
private const val LUT_SLICES = 32

/** 最大 LUT 索引 = `LUT_SLICES - 1`，也就是 shader 里的 `last`。 */
private const val LUT_LAST_INDEX = LUT_SLICES - 1

/**
 * 8bit 通道 -> 0..31 连续 LUT 索引的比例：`index = channel * (31 / 255)`。
 *
 * 这个式子保证 `channel = 255` 正好落在第 31 档（不越界），`channel = 0` 落在第 0 档，
 * 中间的通道值落在两档之间由三线性插值补。GL 侧 `LUT_FRAGMENT_SHADER` 和 CPU 侧
 * [LutTable.lookup] 都读这一个常量，避免两边各写一份走偏。
 *
 * 曾经的错误版本是 `31 / 32`：那是以「归一化到 0..1 的通道」为前提的式子，直接乘 8bit
 * 通道值会把索引放大到 0..247，越界后被 CLAMP_TO_EDGE 钉在最后一列，等于整张 LUT 只用了
 * 边缘一小块。所以这里必须是 255 做分母。
 */
private const val LUT_INDEX_PER_CHANNEL = (LUT_LAST_INDEX.toFloat() / 255.0f)

/**
 * 实时 3D LUT 调色效果。
 *
 * 本文件是自包含实现：EGL + GLES 2.0 全部自己写，不依赖
 * `androidx.camera.effects.opengl.*`（那一层里 `GlProgram` 的构造函数和 `Utils` /
 * `GlProgramCopy` 都是包内可见，app 代码既不能继承也不能调用；`GlRenderer` 虽然
 * 公开但只能整帧 copy + overlay，没法插 LUT）。所以下面照抄
 * `SurfaceProcessorImpl` 的结构（EGL 上下文 / OES 输入纹理 / SurfaceTexture 帧回调 /
 * 输出 Surface 注册与 close 语义），只把渲染程序换成带 LUT 的 shader。
 *
 * 线程模型：所有 GL 调用都跑在 [LutEffect] 传给 `CameraEffect` 的那条单线程 executor 上。
 * [LutSurfaceProcessor.setStyle] 由 UI 线程调用，只写一个 `@Volatile` 字段，下一帧在 GL
 * 线程上被取走并按需重载 LUT 纹理 —— 切风格不重新 bind 用例。
 */
class LutEffect private constructor(
    private val processor: LutSurfaceProcessor,
    executor: Executor,
    onError: (Throwable) -> Unit,
) : CameraEffect(
    TARGETS,
    TRANSFORMATION_ARBITRARY,
    executor,
    processor,
    Consumer<Throwable> { t -> onError(t) },
) {

    companion object {
        private const val TAG = "LutEffect"

        /**
         * ★ 2026-10-04 实机结论：只挂 PREVIEW，**不能**挂 IMAGE_CAPTURE。
         *
         * 原因：SurfaceProcessor 的输入 Surface 只有一份，CameraX 按会话（预览/分析）的
         * 分辨率给（实测 1920×1080）；一旦把 IMAGE_CAPTURE 也交给它，成片就变成这张
         * Surface 的尺寸 —— 实测从 4096×3072（12 MP）掉到 1080×1920（2 MP），
         * 而且给 ImageCapture 显式设 4096 分辨率选择器也拉不回来。
         * 所以成片走「拍完再用同一张 LUT 做一次 CPU 调色」（CameraController.gradeJpeg）。
         * 允许的组合只有 PREVIEW / PREVIEW|VIDEO_CAPTURE / PREVIEW|VIDEO_CAPTURE|IMAGE_CAPTURE。
         */
        private const val TARGETS = CameraEffect.PREVIEW

        /**
         * 选 `TRANSFORMATION_ARBITRARY`（== 0）而不是 `TRANSFORMATION_PASSTHROUGH`。
         *
         * 原因：CameraX 会在 [SurfaceOutput.updateTransformMatrix] 里给出「额外变换」
         * （ViewPort 裁切 / 目标旋转 / 镜像 / 设备级拉伸修正）。本效果不改几何，但必须
         * 把这套变换和 `SurfaceTexture.getTransformMatrix` 的相机变换一起乘进顶点着色器
         * 的纹理坐标，否则预览会错位（PASSTHROUGH 的语义是「处理器忽略变换」，那是给
         * 只做 buffer 共享、不真正绘制的场景用的）。ARBITRARY 恰好表示「处理器自己处理
         * 任意变换」——我们就在 shader 里处理它。
         */
        private const val TRANSFORMATION = TRANSFORMATION_ARBITRARY

        /** 工厂：先造处理器，再把它交给 `CameraEffect` 构造器（处理器要挂在同一个 executor 上）。 */
        @JvmStatic
        fun create(
            context: Context,
            executor: Executor,
            onError: (Throwable) -> Unit,
        ): LutEffect {
            val processor = LutSurfaceProcessor(context, executor, onError)
            return LutEffect(processor, executor, onError)
        }
    }

    /** 切换风格（0=原图 / 1=黑白）：只写 @Volatile 序号，下一帧在 GL 线程生效，不重绑 use case。 */
    fun setStyle(index: Int) {
        processor.setStyle(index)
    }

    /** 风格总数（UI 循环用）。 */
    fun styleCount(): Int = LutStyles.names.size

    /** 释放 GL 资源（onDestroy 时调，可重复调用）。 */
    fun release() {
        processor.release()
    }
}

/**
 * 滤镜的索引 / 名称 / LUT 资源表。
 *
 * 索引就是 UI 上给用户选的序号，必须和下面 `names` 的下标一一对应。
 *
 * 仓库里原来有 7 张胶片模拟的 LUT PNG，归属上只适合自用，已经从仓库移出，
 * 所以现在只剩「原图」和「黑白」两档；LUT 的读表通路还留在代码里，
 * 以后自己产了 LUT 只要把资源放进 `res/raw` 并在 [rawResId] 补分支即可。
 */
object LutStyles {

    /** 原图：不做任何查表，直接输出。 */
    const val ORIGINAL = 0

    /** 黑白：不用 LUT 纹理，shader 里算亮度 + 轻微提对比。 */
    const val BLACK_WHITE = 1

    /** UI 显示用的中文名，下标即风格 index。 */
    val names: List<String> = listOf(
        "原图",
        "黑白",
    )

    /**
     * 返回该风格对应的 `res/raw` LUT 资源；[ORIGINAL] 和 [BLACK_WHITE] 没有 LUT，返回 null。
     *
     * 目前仓库里不带 LUT 图片，所以恒为 null；调用方本来就按"读不到 LUT 就直通"处理。
     */
    fun rawResId(index: Int): Int? = null
}

/**
 * 成片（照片）的 CPU 调色：把和预览 shader **同一套**数学搬到 `IntArray` 像素上。
 *
 * 为什么需要它：`CameraEffect` 只要把 `IMAGE_CAPTURE` 纳进 targets，成片分辨率就会掉到
 * 会话分辨率（实机 1920x1080），所以成片不在 GL 通路里做，改由拍照后在后台跑一遍。
 *
 * 与 `LUT_FRAGMENT_SHADER` / `BW_FRAGMENT_SHADER` 的逐项对应：
 *
 * | shader | 这里 |
 * | --- | --- |
 * | `p = c * (31.0 / 255.0)`（c 为归一化通道） | [lookup] 里同一个常量 [LUT_INDEX_PER_CHANNEL] |
 * | `r0 = floor(p)`, `r1 = min(r0+1, 31)` | 同名的整数版本（NEAREST 采样下标就是整数） |
 * | `mix(mix(c00,c10,fr), mix(c02,c12,fr), fg)` 再按 `fb` 插值 | [lookup] 的 8 邻居三线性 |
 * | `uLutScale` 为 0 时 `mix(src, lut, 0)` → 原样输出 | [apply] 里 `scale <= 0` 直接返回 |
 * | BW：`dot(src, vec3(0.2126,0.7152,0.0722))` → `clamp((y-0.5)*1.10+0.5, 0, 1)` | [processChunk] 的 BW 分支 |
 *
 * 两个刻意的取舍：
 * 1. **不做 sRGB → 线性**：图上 `LINEARIZE_SRGB = false`（见 `LutEffect` companion 里的说明），
 *    CPU 侧就跟着不做。哪天开关翻回 true，这里也要同步加 `srgbToLinear`。
 * 2. **alpha 原样保留**：shader 输出不透明，成片这里不替调用方决定 alpha 语义。
 */
object LutPostProcess {

    /**
     * 把 [pixels]（ARGB_8888 打包，长度 = `width * height`）在原地调色。
     *
     * 风格不可用（越界 / [LutStyles.ORIGINAL] / 贴图读不出来）时按直通处理，**不抛异常也不改像素**。
     *
     * @param context 读 `res/raw` 里的 LUT PNG；内部只取 applicationContext。
     * @param pixels ARGB_8888 像素，原地修改。
     * @param width 图片宽，`pixels.size` 必须等于 `width * height`。
     * @param height 图片高。
     * @param styleIndex 风格下标，见 [LutStyles]。
     */
    fun apply(context: Context, pixels: IntArray, width: Int, height: Int, styleIndex: Int) {
        val table = tableFor(context, styleIndex)
        apply(table, pixels, width, height, styleIndex)
    }

    /**
     * 已经拿到表（或刻意要直通）时用这个版本，不碰 `Context`。
     *
     * [table] 传 null 表示直通；[styleIndex] 只用来选 BW 分支和判断 [LutStyles.ORIGINAL]。
     */
    internal fun apply(
        table: LutTable?,
        pixels: IntArray,
        width: Int,
        height: Int,
        styleIndex: Int,
    ) {
        val total = width * height
        if (total <= 0 || pixels.size < total) return
        if (styleIndex == LutStyles.ORIGINAL) return
        if (table == null && styleIndex != LutStyles.BLACK_WHITE) return

        val threads = minOf(4, Runtime.getRuntime().availableProcessors())
        if (threads <= 1) {
            processChunk(table, pixels, width, height, styleIndex, 0, height)
            return
        }

        val failure = java.util.concurrent.atomic.AtomicReference<Throwable?>(null)
        val workers = ArrayList<Thread>(threads - 1)
        val rowsPerChunk = (height + threads - 1) / threads
        var start = 0
        while (start < height) {
            val end = minOf(start + rowsPerChunk, height)
            val from = start
            val to = end
            if (to >= height) {
                // 最后一段留在当前线程跑，省一次线程创建。
                processChunk(table, pixels, width, height, styleIndex, from, to)
            } else {
                val worker = Thread({
                    try {
                        processChunk(table, pixels, width, height, styleIndex, from, to)
                    } catch (t: Throwable) {
                        failure.compareAndSet(null, t)
                    }
                }, "LutPostProcess-$from")
                worker.isDaemon = true
                workers.add(worker)
                worker.start()
            }
            start = end
        }

        var interrupted: InterruptedException? = null
        for (w in workers) {
            try {
                w.join()
            } catch (e: InterruptedException) {
                interrupted = e
            }
        }
        if (interrupted != null) {
            Thread.currentThread().interrupt()
            throw interrupted
        }
        // 任一分片失败就整体失败，绝不“跑了一半还算成功”。
        failure.get()?.let { throw it }
    }

    /** 处理 `[fromRow, toRow)` 的行区间。 */
    private fun processChunk(
        table: LutTable?,
        pixels: IntArray,
        width: Int,
        height: Int,
        styleIndex: Int,
        fromRow: Int,
        toRow: Int,
    ) {
        val isGray = styleIndex == LutStyles.BLACK_WHITE
        val lut = if (isGray) null else table ?: return
        for (y in fromRow until toRow) {
            val rowStart = y * width
            for (x in 0 until width) {
                val index = rowStart + x
                val argb = pixels[index]
                val r = (argb ushr 16) and 0xFF
                val g = (argb ushr 8) and 0xFF
                val b = argb and 0xFF
                // 0xFF000000 那一档是 alpha：两条分支都原样保留。
                val alpha = argb and 0xFF000000.toInt()
                if (isGray) {
                    // 对齐 BW_FRAGMENT_SHADER：Rec.709 亮度 + 围着中灰 0.5 的轻微提对比。
                    val yLin = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f
                    val contrasted = clamp01((yLin - 0.5f) * 1.10f + 0.5f)
                    val v = round255(contrasted)
                    pixels[index] = alpha or (v shl 16) or (v shl 8) or v
                } else {
                    // 查表返回打包好的 0xRRGGBB。**不能**把结果放在 table 的共享字段里：
                    // 这里是多线程按行分片跑的，共享字段会让别的线程在 lookup 和读取之间
                    // 插进来，把通道串掉。
                    pixels[index] = alpha or lut!!.lookup(r, g, b)
                }
            }
        }
    }

    /**
     * 该风格的 LUT 表，按 style 缓存。解码失败 / 尺寸不对返回 null，由调用方按直通处理。
     *
     * 缓存用「先查 volatile，再进同步块复查」的双检锁：需要保的是
     * monitor 释放 → volatile 写入 → monitor 获取 这条链，表内容才安全可见。
     */
    private fun tableFor(context: Context, styleIndex: Int): LutTable? {
        if (styleIndex == LutStyles.ORIGINAL || styleIndex == LutStyles.BLACK_WHITE) return null
        val resId = LutStyles.rawResId(styleIndex) ?: return null
        cache[styleIndex]?.let { return it }
        synchronized(cacheLock) {
            cache[styleIndex]?.let { return it }
            val table = try {
                loadTable(context, resId)
            } catch (e: Exception) {
                Log.w(TAG, "读取 LUT 失败 style=$styleIndex resId=$resId: ${e.message}", e)
                null
            } ?: return null
            cache[styleIndex] = table
            return table
        }
    }

    /** 解 PNG 成 [LutTable]。非 `ARGB_8888` 会先转成它，保证 [IntArray] 的通道序确定。 */
    private fun loadTable(context: Context, resId: Int): LutTable? {
        val decoded = context.applicationContext.resources.openRawResource(resId).use { input ->
            BitmapFactory.decodeStream(input)
        } ?: run {
            Log.w(TAG, "LUT 解码失败 resId=$resId")
            return null
        }
        val bitmap = if (decoded.config == Bitmap.Config.ARGB_8888) {
            decoded
        } else {
            val converted = decoded.copy(Bitmap.Config.ARGB_8888, false)
            decoded.recycle()
            converted
        }
        try {
            if (bitmap.width != LUT_TEX_WIDTH || bitmap.height != LUT_TEX_HEIGHT) {
                Log.w(
                    TAG,
                    "LUT 尺寸异常: ${bitmap.width}x${bitmap.height}，期望 " +
                        "${LUT_TEX_WIDTH}x${LUT_TEX_HEIGHT}，该风格按直通处理",
                )
                return null
            }
            val pixels = IntArray(LUT_TEX_WIDTH * LUT_TEX_HEIGHT)
            bitmap.getPixels(
                pixels,
                0,
                LUT_TEX_WIDTH,
                0,
                0,
                LUT_TEX_WIDTH,
                LUT_TEX_HEIGHT,
            )
            return LutTable.fromPixels(pixels)
        } finally {
            bitmap.recycle()
        }
    }

    private const val TAG = "LutEffect"

    /** style -> 表。只在 [cacheLock] 里写，读取先走 volatile。 */
    @Volatile
    private var cache: Array<LutTable?> = arrayOfNulls(LutStyles.names.size)

    private val cacheLock = Any()
}

/**
 * 一张已经摊平成 32x32x32 的 LUT，取值 0..255。
 *
 * `table[g][r][b][c]`：和纹理布局 `x = b*32 + r`、`y = g` 完全一致 —— 纹理列方向红最快，
 * 所以摊平时红在最低位、蓝在最高位。
 */
internal class LutTable private constructor(private val data: FloatArray) {

    /**
     * 三线性采样一个像素，返回打包好的 `0xRRGGBB`。
     *
     * 返回值而不是写字段：`LutPostProcess` 是按行分片多线程跑的，一个 `LutTable` 会被多条
     * 线程共享；如果把结果放在实例字段里，「查完再读」之间就会被别的线程插进来串通道。
     * 返回值也顺带省掉每像素的数组分配。
     *
     * 与 shader 的 [LUT_INDEX_PER_CHANNEL] 映射和 8 邻居权重逐项一致：shader 那边
     * 加了 0.5 纹素偏移，但 NEAREST 采样下那 0.5 只会被 `floor` 丢掉，所以这里直接用整数下标。
     */
    fun lookup(r: Int, g: Int, b: Int): Int {
        // 8bit 通道必然 ≤255；越界输入在这里夹住，避免算出 table 外的下标。
        val cr = if (r < 0) 0 else if (r > 255) 255 else r
        val cg = if (g < 0) 0 else if (g > 255) 255 else g
        val cb = if (b < 0) 0 else if (b > 255) 255 else b
        // 8bit 通道 -> 0..31 的连续索引，再夹进表范围（越界输入也要安全）。
        val pr = clampCoord(cr * LUT_INDEX_PER_CHANNEL)
        val pg = clampCoord(cg * LUT_INDEX_PER_CHANNEL)
        val pb = clampCoord(cb * LUT_INDEX_PER_CHANNEL)

        // floor：NEAREST 采样的整数下标就是格点。每维的下标只可能是 0..31。
        val r0 = pr.toInt()
        val g0 = pg.toInt()
        val b0 = pb.toInt()
        val r1 = minOf(r0 + 1, LUT_LAST_INDEX)
        val g1 = minOf(g0 + 1, LUT_LAST_INDEX)
        val b1 = minOf(b0 + 1, LUT_LAST_INDEX)
        val fr = pr - r0
        val fg = pg - g0
        val fb = pb - b0

        var accR = 0f
        var accG = 0f
        var accB = 0f
        for (dg in 0..1) {
            val gg = if (dg == 0) g0 else g1
            val wg = if (dg == 0) 1f - fg else fg
            if (wg == 0f) continue
            for (dr in 0..1) {
                val rr = if (dr == 0) r0 else r1
                val wr = if (dr == 0) 1f - fr else fr
                if (wr == 0f) continue
                for (db in 0..1) {
                    val bb = if (db == 0) b0 else b1
                    val wb = if (db == 0) 1f - fb else fb
                    if (wb == 0f) continue
                    val w = wr * wg * wb
                    val base = ((gg * LUT_SLICES + rr) * LUT_SLICES + bb) * 3
                    accR += w * data[base]
                    accG += w * data[base + 1]
                    accB += w * data[base + 2]
                }
            }
        }
        val outR = clamp255Round(accR)
        val outG = clamp255Round(accG)
        val outB = clamp255Round(accB)
        return (outR shl 16) or (outG shl 8) or outB
    }
    companion object {
        /** 从 1024x32 的 ARGB_8888 像素摊平。 */
        fun fromPixels(pixels: IntArray): LutTable {
            val data = FloatArray(LUT_SLICES * LUT_SLICES * LUT_SLICES * 3)
            for (b in 0 until LUT_SLICES) {
                for (r in 0 until LUT_SLICES) {
                    val column = b * LUT_SLICES + r
                    for (g in 0 until LUT_SLICES) {
                        val argb = pixels[g * LUT_TEX_WIDTH + column]
                        val base = ((g * LUT_SLICES + r) * LUT_SLICES + b) * 3
                        data[base] = ((argb ushr 16) and 0xFF).toFloat()
                        data[base + 1] = ((argb ushr 8) and 0xFF).toFloat()
                        data[base + 2] = (argb and 0xFF).toFloat()
                    }
                }
            }
            return LutTable(data)
        }
    }
}

/** `clamp(c, 0, 1)`。 */
private fun clamp01(v: Float): Float = if (v < 0f) 0f else if (v > 1f) 1f else v

/**
 * LUT 连续索引夹到 `[0, 31]`。
 *
 * 上面的 `index = channel * (31/255)` 本身就不会越界，这里是纯防御：任何意外的输入都不该
 * 让下标算到 32³ 表外面去。
 */
private fun clampCoord(v: Float): Float =
    if (v <= 0f) 0f else if (v >= LUT_LAST_INDEX.toFloat()) LUT_LAST_INDEX.toFloat() else v

/** `clamp(c, 0, 255)`。 */
private fun clamp255(v: Float): Float = if (v < 0f) 0f else if (v > 255f) 255f else v

/** LUT 表里的值本来就在 0..255 量纲上，夹住后四舍五入到整数。 */
private fun clamp255Round(v: Float): Int = Math.round(clamp255(v))

/**
 * 归一化浮点结果 → 8bit 整数。
 *
 * GPU 把 float 写进 RGBA8 目标时按 `round(clamp(c,0,1) * 255)` 取整，这里用同一个规则，
 * 所以 CPU 侧 BW 分支的灰度和 shader 输出能对上。
 */
private fun round255(normalized: Float): Int = Math.round(clamp01(normalized) * 255.0f)

/**
 * 真正干活的 `SurfaceProcessor` 实现。
 *
 * 生命周期与 [androidx.camera.effects.internal.SurfaceProcessorImpl] 对齐：
 * 输入用 OES 纹理 + `SurfaceTexture`，输出用 `SurfaceOutput.getSurface` 拿到的 Surface，
 * 帧到来时 `updateTexImage()` → 取变换矩阵 → 画到输出 Surface → `eglSwapBuffers`。
 *
 * @param context 用来读 `res/raw` 里的 LUT PNG（只保留 applicationContext）。
 * @param executor CameraX 调用本处理器的单线程 executor，GL 上下文就建在这条线程上。
 * @param onError 出错回调（GL 初始化失败 / LUT 载入失败等），保证不抛到调用方。
 */
class LutSurfaceProcessor internal constructor(
    context: Context,
    private val executor: Executor,
    private val onError: (Throwable) -> Unit,
) : SurfaceProcessor, SurfaceTexture.OnFrameAvailableListener {

    private val appContext: Context = context.applicationContext
    private val renderer = GlLutRenderer(appContext) { t -> report(t) }

    /** UI 线程写、GL 线程读：请求的风格；下一帧与已载入的风格对比，不同就换 LUT。 */
    @Volatile
    private var requestedStyle: Int = LutStyles.ORIGINAL

    /** 已 release 后不再接任何活，`release()` 也保证幂等。 */
    @Volatile
    private var released: Boolean = false

    /**
     * 切换风格。线程安全：只写 `@Volatile` 字段，GL 线程在下一帧取走。
     * 不会重新绑定用例，所以预览不闪。
     */
    fun setStyle(index: Int) {
        val safe = if (index in LutStyles.names.indices) index else LutStyles.ORIGINAL
        requestedStyle = safe
        renderer.requestStyle(safe)
    }

    /** 当前请求的风格（UI 回显用，不保证 GL 侧已经生效）。 */
    fun style(): Int = requestedStyle

    override fun onInputSurface(request: SurfaceRequest) {
        if (released) {
            // CameraX 关掉相机时仍在投递请求，明确告诉它不会提供 Surface。
            request.willNotProvideSurface()
            return
        }
        runOnGlThread(GlTask { renderer.onInputSurface(request, requestedStyle) })
    }

    override fun onOutputSurface(output: SurfaceOutput) {
        if (released) {
            output.close()
            return
        }
        runOnGlThread(GlTask { renderer.onOutputSurface(output) })
    }

    override fun onFrameAvailable(surfaceTexture: SurfaceTexture) {
        if (released) return
        // 回调注册在 GL 线程的 Handler 上，所以这里已经在 GL 线程。
        runOnGlThread(GlTask { renderer.onFrameAvailable(surfaceTexture) })
    }

    /**
     * 释放全部 GL 资源。幂等：重复调用是空操作（MainActivity 在 onDestroy 里调）。
     */
    fun release() {
        if (released) return
        released = true
        // 注意：released 只是「不再接新活」的闸门，renderer.release() 必须真的投出去，
        // 不能也去看这个标志位，否则资源永远不释放。
        runOnGlThread(GlTask {
            renderer.release()
            renderer.shutdown()
        })
    }

    // --- 内部 ---

    private fun runOnGlThread(task: GlTask) {
        try {
            if (renderer.isGlThread()) {
                task.run()
            } else {
                renderer.post(task)
            }
        } catch (t: Throwable) {
            report(t)
        }
    }

    private fun report(t: Throwable) {
        Log.e(TAG, "LUT 效果出错: ${t.message}", t)
        try {
            onError(t)
        } catch (e: Exception) {
            Log.e(TAG, "onError 回调本身抛异常，忽略", e)
        }
    }

    companion object {
        private const val TAG = "LutEffect"
    }
}

/** 可抛异常的 GL 任务；在 GL 线程上执行，异常统一转成 `onError`。 */
internal fun interface GlTask {
    @Throws(Exception::class)
    fun run()
}

/**
 * 自包含的 EGL + GLES2 渲染器：把 OES 输入纹理经 3D LUT 调色后画到输出 Surface。
 *
 * 除了 LUT/黑白两个 fragment shader 之外，EGL 的建/毁、SurfaceTexture 帧回调、输出
 * Surface 的注册与 close 语义都照 `SurfaceProcessorImpl` 的做法来。
 *
 * 只有一条 GL 线程（`handlerThread`），所有方法都必须在它上面调用。
 */
private class GlLutRenderer(
    private val context: Context,
    private val onError: (Throwable) -> Unit,
) : SurfaceTexture.OnFrameAvailableListener {

    // --- 线程 ---

    private val handlerThread = HandlerThread("LutEffect-GL")
    private val handler: Handler
    private val glThread: Thread

    init {
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        glThread = handlerThread.looper.thread
    }

    fun isGlThread(): Boolean = Thread.currentThread() === glThread

    fun post(task: GlTask) {
        handler.post {
            try {
                task.run()
            } catch (t: Throwable) {
                onError(t)
            }
        }
    }

    fun shutdown() {
        handlerThread.quitSafely()
    }

    // --- EGL ---

    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglConfig: EGLConfig? = null
    private var tempSurface: EGLSurface? = null

    /** 当前绑定的输出 Surface 对应的 EGLSurface（同时记尺寸，画的时候设 viewport）。 */
    private var currentOut: OutputEgl? = null

    // --- 输入 ---

    private var inputTextureId = 0
    private var inputSurfaceTexture: SurfaceTexture? = null

    /** 输入分辨率，仅当查询不到输出 Surface 尺寸时用作 viewport 兜底。 */
    private var inputWidth = 0
    private var inputHeight = 0

    // --- 输出 ---

    /** 正在用的输出（`SurfaceOutput` + 它的 `Surface` + EGLSurface）。 */
    private var outputPair: OutputPair? = null

    /** 收到 EVENT_REQUEST_CLOSE、下一帧丢掉 EGLSurface 的输出。 */
    private val pendingClose = ArrayList<SurfaceOutput>()
    private val closedOutputs = LinkedHashSet<SurfaceOutput>()

    // --- GL 对象 ---

    private var lutProgram: GlProgram? = null
    private var bwProgram: GlProgram? = null
    private var lutTextureId = 0

    /** LUT 上传连续失败次数（>=3 就放弃该风格，避免每帧重试）。 */
    private var lutFailCount = 0

    /** 当前 LUT 纹理对应的风格 index；`-1` 表示还没有载入任何 LUT。 */
    private var loadedLutStyle = -1

    private var released = false

    /** 渲染器自己记一份请求风格，供帧回调使用（UI 线程写、GL 线程读）。 */
    @Volatile
    private var requestedStyle: Int = LutStyles.ORIGINAL

    // 复用矩阵，避免每帧分配
    private val cameraMatrix = FloatArray(16)
    private val outputMatrix = FloatArray(16)

    /**
     * UI 线程调用：只写一个 `@Volatile` 字段，下一帧在 GL 线程上生效。
     * 不重建任何 GL 对象、更不重新绑定用例，所以切风格不会闪。
     */
    fun requestStyle(index: Int) {
        requestedStyle = index
    }

    // ==================================================================
    // 公开给 LutSurfaceProcessor 的入口（均在 GL 线程调用）
    // ==================================================================

    fun onInputSurface(request: SurfaceRequest, requestedStyle: Int) {
        if (released) {
            request.willNotProvideSurface()
            return
        }
        requestStyle(requestedStyle)
        initGlIfNeeded()
        if (inputTextureId == 0) {
            // GL 初始化失败（已通过 onError 上报过），不能拿 0 号纹理去建 SurfaceTexture。
            request.willNotProvideSurface()
            return
        }
        val previous = inputSurfaceTexture
        if (previous != null) {
            previous.setOnFrameAvailableListener(null)
            previous.release()
            inputSurfaceTexture = null
        }

        val resolution = request.resolution
        val surfaceTexture = SurfaceTexture(inputTextureId)
        surfaceTexture.setDefaultBufferSize(resolution.width, resolution.height)
        val surface = Surface(surfaceTexture)

        inputSurfaceTexture = surfaceTexture
        inputWidth = resolution.width
        inputHeight = resolution.height

        request.provideSurface(surface, executor) { result ->
            // CameraX 用完了 / 取消了这次请求：按约定释放 SurfaceTexture 和 Surface。
            Log.d(TAG, "输入 Surface 结束: code=${result.resultCode}")
            surfaceTexture.setOnFrameAvailableListener(null)
            surfaceTexture.release()
            surface.release()
            if (inputSurfaceTexture === surfaceTexture) {
                inputSurfaceTexture = null
            }
        }
        surfaceTexture.setOnFrameAvailableListener(this, handler)

        applyStyleIfNeeded(requestedStyle)
    }

    fun onOutputSurface(output: SurfaceOutput) {
        if (released) {
            if (closedOutputs.add(output)) output.close()
            return
        }
        initGlIfNeeded()

        val pair = outputPair
        if (pair != null && pair.surfaceOutput === output) {
            // 同一个 SurfaceOutput 又给了一次，直接复用。
            return
        }
        // CameraX 可能在让我们 close 旧输出之前就给来新的：先把旧的挪进待关闭队列，
        // 等这一帧画完再真正 close（它内部的 EGLSurface 也要销毁）。
        if (pair != null) {
            if (closedOutputs.add(pair.surfaceOutput)) {
                pendingClose.add(pair.surfaceOutput)
            }
        }

        val surface = output.getSurface(executor) { event ->
            if (event.eventCode == SurfaceOutput.Event.EVENT_REQUEST_CLOSE) {
                Log.d(TAG, "输出 Surface 请求关闭")
                if (closedOutputs.add(output)) {
                    output.close()
                    // close() 之后 CameraX 才会真的 release Surface，这时才能销毁 EGLSurface。
                    pendingClose.add(output)
                }
            }
        }
        outputPair = OutputPair(output, surface)
        Log.d(TAG, "输出 Surface 已注册 size=${output.size}")
    }

    override fun onFrameAvailable(surfaceTexture: SurfaceTexture) {
        if (released) return
        val pair = outputPair
        if (pair == null) {
            // 输出还没准备好，这一帧丢掉。
            return
        }

        try {
            // ★ 2026-10-04 真机定位的关键一行：SurfaceTexture.updateTexImage() 会把
            //   相机 OES 纹理绑到**当前激活的纹理单元**上。上一帧 drawFrame 结束时激活
            //   单元停在 GL_TEXTURE1（bindLutTexture 留下的），于是运行中切风格时
            //   LUT 的 TEXTURE_2D 和相机的 EXTERNAL_OES 纹理挤在同一个单元上，
            //   紧接着的 texImage2D 会让相机帧流停掉（现象：预览全黑、之后
            //   再也不出帧，日志里只有 SurfaceTexture 的 "clearing GL error: 0x502"）。
            //   固定先切回 unit 0，OES 永远只绑 unit 0、LUT 只用 unit 1，互不干扰。
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            surfaceTexture.updateTexImage()
            surfaceTexture.getTransformMatrix(cameraMatrix)

            // CameraX 的 updateTransformMatrix 返回的矩阵已经含相机变换，直接交给 drawFrame。
            pair.surfaceOutput.updateTransformMatrix(outputMatrix, cameraMatrix)

            applyStyleIfNeeded(requestedStyle)

            drawFrame(pair, outputMatrix, surfaceTexture.timestamp)
        } catch (e: Exception) {
            Log.e(TAG, "渲染一帧失败，丢弃该帧: ${e.message}", e)
            onError(e)
        }
    }

    fun release() {
        if (released) return
        released = true
        try {
            inputSurfaceTexture?.setOnFrameAvailableListener(null)
            inputSurfaceTexture?.release()
            inputSurfaceTexture = null

            outputPair?.let { pair ->
                if (closedOutputs.add(pair.surfaceOutput)) {
                    pair.surfaceOutput.close()
                }
                unregisterEglSurface(pair.surface)
            }
            outputPair = null
            for (o in pendingClose) {
                if (closedOutputs.add(o)) o.close()
            }
            pendingClose.clear()

            if (lutTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(lutTextureId), 0)
                lutTextureId = 0
            }
            loadedLutStyle = -1
            lutProgram?.release()
            lutProgram = null
            bwProgram?.release()
            bwProgram = null
            if (inputTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(inputTextureId), 0)
                inputTextureId = 0
            }
            releaseEgl()
        } catch (e: Exception) {
            Log.w(TAG, "释放 GL 资源时出错: ${e.message}", e)
        }
    }

    // ==================================================================
    // EGL
    // ==================================================================

    private fun initGlIfNeeded() {
        if (eglDisplay != null) return
        try {
            createEgl()
            // 输入纹理：外部纹理 + NEAREST，1:1 拷贝不需要过滤。
            val ids = IntArray(1)
            GLES20.glGenTextures(1, ids, 0)
            checkGl("glGenTextures")
            inputTextureId = ids[0]
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, inputTextureId)
            GLES20.glTexParameterf(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST.toFloat()
            )
            GLES20.glTexParameterf(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST.toFloat()
            )
            GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE
            )
            GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE
            )
            checkGl("配置输入纹理")

            lutProgram = GlProgram(LUT_VERTEX_SHADER, LUT_FRAGMENT_SHADER, TEX_SIZE)
            bwProgram = GlProgram(LUT_VERTEX_SHADER, BW_FRAGMENT_SHADER, TEX_SIZE)
            // uTexSize 是「一个纹素的归一化尺寸」(1/1024, 1/32)，只跟 LUT 纹理尺寸有关，
            // 建 program 时设一次即可。不设的话默认 (0,0)，8 次采样会全落在同一个纹素上。
            lutProgram?.setTexSize(LUT_TEX_WIDTH, LUT_TEX_HEIGHT)
            // 先按「无 LUT」建一张恒等纹理，任何风格切换前都能安全绘制。
            applyStyle(LutStyles.ORIGINAL)
        } catch (t: Throwable) {
            Log.e(TAG, "GL 初始化失败: ${t.message}", t)
            release()
            onError(t)
        }
    }

    private fun createEgl() {
        val display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (display == null || display == EGL14.EGL_NO_DISPLAY) {
            throw IllegalStateException("eglGetDisplay 失败")
        }
        val version = IntArray(2)
        if (!EGL14.eglInitialize(display, version, 0, version, 1)) {
            throw IllegalStateException(
                "eglInitialize 失败: 0x" + Integer.toHexString(EGL14.eglGetError())
            )
        }
        eglDisplay = display

        val attribs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGLExt.EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT or EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(display, attribs, 0, configs, 0, 1, numConfigs, 0) ||
            numConfigs[0] <= 0
        ) {
            throw IllegalStateException(
                "eglChooseConfig 失败: 0x" + Integer.toHexString(EGL14.eglGetError())
            )
        }
        val config = configs[0] ?: throw IllegalStateException("eglChooseConfig 没有可用配置")
        eglConfig = config

        val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        val context = EGL14.eglCreateContext(display, config, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)
        if (context == null || context == EGL14.EGL_NO_CONTEXT) {
            throw IllegalStateException(
                "eglCreateContext 失败: 0x" + Integer.toHexString(EGL14.eglGetError())
            )
        }
        eglContext = context

        // 1x1 pbuffer：还没有输出 Surface 时也要有个 current 的 EGLSurface。
        val pbufferAttribs = intArrayOf(
            EGL14.EGL_WIDTH, 1,
            EGL14.EGL_HEIGHT, 1,
            EGL14.EGL_NONE,
        )
        val pbuffer = EGL14.eglCreatePbufferSurface(display, config, pbufferAttribs, 0)
        if (pbuffer == null || pbuffer == EGL14.EGL_NO_SURFACE) {
            throw IllegalStateException(
                "eglCreatePbufferSurface 失败: 0x" + Integer.toHexString(EGL14.eglGetError())
            )
        }
        tempSurface = pbuffer
        if (!EGL14.eglMakeCurrent(display, pbuffer, pbuffer, context)) {
            throw IllegalStateException(
                "eglMakeCurrent 失败: 0x" + Integer.toHexString(EGL14.eglGetError())
            )
        }
    }

    /**
     * 上传/删除 GL 对象前确保当前线程有 current 的 EGL 上下文。
     * 有输出 Surface 就绑它，否则回落到 1x1 pbuffer（`tempSurface`）。
     */
    private fun ensureGlCurrent(): Boolean {
        val display = eglDisplay ?: return false
        val context = eglContext ?: return false
        if (EGL14.eglGetCurrentContext() == context) return true
        val out = currentOut
        val surface = when {
            out != null && out.eglSurface != EGL14.EGL_NO_SURFACE -> out.eglSurface
            tempSurface != null && tempSurface != EGL14.EGL_NO_SURFACE -> tempSurface
            else -> return false
        }
        val ok = EGL14.eglMakeCurrent(display, surface, surface, context)
        if (!ok) {
            Log.w(TAG, "eglMakeCurrent 失败: 0x" + Integer.toHexString(EGL14.eglGetError()))
        }
        return ok
    }

    private fun releaseEgl() {        val display = eglDisplay ?: return
        EGL14.eglMakeCurrent(
            display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT
        )
        currentOut?.let { EGL14.eglDestroySurface(display, it.eglSurface) }
        currentOut = null
        tempSurface?.let { EGL14.eglDestroySurface(display, it) }
        tempSurface = null
        eglContext?.let { EGL14.eglDestroyContext(display, it) }
        eglContext = null
        EGL14.eglTerminate(display)
        EGL14.eglReleaseThread()
        eglDisplay = null
        eglConfig = null
    }

    /** 为输出 Surface 建/取 EGLSurface；失败返回 null（这一帧跳过，不炸）。 */
    private fun ensureOutputEgl(surface: Surface): OutputEgl? {
        val existing = currentOut
        if (existing != null && existing.surface === surface && existing.eglSurface != EGL14.EGL_NO_SURFACE) {
            return existing
        }
        existing?.let { unregisterEglSurface(it.surface) }
        val display = eglDisplay ?: return null
        val config = eglConfig ?: return null
        return try {
            val eglSurface = EGL14.eglCreateWindowSurface(
                display, config, surface, intArrayOf(EGL14.EGL_NONE), 0
            )
            if (eglSurface == null || eglSurface == EGL14.EGL_NO_SURFACE) {
                Log.w(TAG, "eglCreateWindowSurface 失败: 0x" + Integer.toHexString(EGL14.eglGetError()))
                null
            } else {
                val w = querySurface(eglSurface, EGL14.EGL_WIDTH)
                val h = querySurface(eglSurface, EGL14.EGL_HEIGHT)
                OutputEgl(surface, eglSurface, w, h).also { currentOut = it }
            }
        } catch (e: Exception) {
            Log.w(TAG, "创建 EGLSurface 异常: ${e.message}", e)
            null
        }
    }

    private fun unregisterEglSurface(surface: Surface) {
        val current = currentOut ?: return
        if (current.surface !== surface) return
        val display = eglDisplay
        if (display != null) {
            if (isCurrent(current.eglSurface)) {
                tempSurface?.let { EGL14.eglMakeCurrent(display, it, it, eglContext) }
            }
            EGL14.eglDestroySurface(display, current.eglSurface)
        }
        currentOut = null
    }

    private fun isCurrent(eglSurface: EGLSurface): Boolean =
        EGL14.eglGetCurrentSurface(EGL14.EGL_DRAW) == eglSurface

    private fun querySurface(eglSurface: EGLSurface, what: Int): Int {
        val value = IntArray(1)
        EGL14.eglQuerySurface(eglDisplay, eglSurface, what, value, 0)
        return value[0]
    }

    // ==================================================================
    // 渲染
    // ==================================================================

    private fun drawFrame(
        pair: OutputPair,
        outputTransform: FloatArray,
        timestampNs: Long,
    ) {
        if (released) return
        val out = ensureOutputEgl(pair.surface) ?: return
        val display = eglDisplay ?: return
        val context = eglContext ?: return

        if (!EGL14.eglMakeCurrent(display, out.eglSurface, out.eglSurface, context)) {
            Log.w(TAG, "eglMakeCurrent 失败: 0x" + Integer.toHexString(EGL14.eglGetError()))
            return
        }

        // 直接喂 outputTransform，**不要再乘 cameraTransform**。
        // SurfaceOutputImpl.updateTransformMatrix(updated, input) 内部是
        //   Matrix.multiplyMM(updated, 0, input, 0, additionalTransform, 0)
        // 调用方传进来的 input 就是 surfaceTexture.getTransformMatrix() 拿到的相机矩阵，
        // 所以 updated 已经 = 相机变换 × 额外变换，相机变换只在里面应用了一次
        // （见 D:\spider\.cache\cam-src\core\androidx\camera\core\processing\SurfaceOutputImpl.java:283-284）。
        // androidx.camera.effects.internal.SurfaceProcessorImpl 也正是把这份矩阵单独喂给渲染器：
        //   surfaceOutput.updateTransformMatrix(mSurfaceTransform, mTextureTransform);
        //   mGlRenderer.renderInputToSurface(timestamp, mSurfaceTransform, surface);
        // 之前这里又乘了一次 cameraTransform，相机旋转被应用两遍，预览就多转了 90°。

        val width = if (out.width > 0) out.width else inputWidth
        val height = if (out.height > 0) out.height else inputHeight
        GLES20.glViewport(0, 0, width, height)
        GLES20.glScissor(0, 0, width, height)
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_CULL_FACE)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        val requested = requestedStyleOfLastApply
        val program = if (requested == LutStyles.BLACK_WHITE) bwProgram else lutProgram
        if (program == null) return

        program.use()
        program.setMatrix(outputTransform)
        program.bindInputTexture(inputTextureId)
        if (program !== bwProgram) {
            program.bindLutTexture(lutTextureId)
            program.setScale(if (lutTextureId == 0) 0f else 1f)
        }
        program.setSrgbToLinear(LINEARIZE_SRGB)
        program.draw()

        val error = GLES20.glGetError()
        if (error != GLES20.GL_NO_ERROR) {
            Log.w(TAG, "绘制出现 GL error: 0x" + Integer.toHexString(error))
        }

        EGLExt.eglPresentationTimeANDROID(display, out.eglSurface, timestampNs)
        if (!EGL14.eglSwapBuffers(display, out.eglSurface)) {
            // 失败一般是 Surface 已经不可用（切页面/退后台），丢掉它，下一帧重建。
            Log.w(TAG, "eglSwapBuffers 失败: 0x" + Integer.toHexString(EGL14.eglGetError()))
            unregisterEglSurface(out.surface)
        }

        // 帧画完了，现在可以安全关闭待关闭的输出。
        flushPendingClose()
    }

    /** 当前这一帧实际用的风格（`applyStyleIfNeeded` 记录）。 */
    private var requestedStyleOfLastApply = LutStyles.ORIGINAL

    // ==================================================================
    // LUT
    // ==================================================================

    private fun applyStyleIfNeeded(requestedStyle: Int) {
        if (requestedStyle == loadedLutStyle) return
        applyStyle(requestedStyle)
    }

    /**
     * 载入该风格的 LUT。载入失败只记 warning —— 纹理置 0，shader 走「恒等 LUT」路径，
     * 也就是原样输出，绝不因为一张 PNG 读不出来就让预览黑屏。
     */
    private fun applyStyle(requestedStyle: Int) {
        // ★ 2026-10-04 实测：刚重建出来的效果实例偶尔会在上下文还没 current 时就来上传，
        //   glGenTextures 直接吃 GL error 0x502（该风格静默退回原图，画面不黑但没效果）。
        //   这里先保证有 current 的 EGL 上下文；拿不到就跳过，下一帧再试。
        if (!ensureGlCurrent()) {
            Log.w(TAG, "LUT 载入跳过（style=$requestedStyle）：当前没有可用的 GL 上下文")
            return
        }
        // ★ 先把之前遗留的 GL 错误读干净：checkGl 读的是「最早未读的那个错误」，
        //   残留错误会让 glGenTextures 的检查误报，把整个 LUT 载入判成失败
        //   （实测切风格时约一半概率因此静默退回原图）。
        while (GLES20.glGetError() != GLES20.GL_NO_ERROR) {
            // drain
        }
        requestedStyleOfLastApply = requestedStyle
        val resId = LutStyles.rawResId(requestedStyle)
        if (resId == null) {
            // 原图 / 黑白 / 越界：不需要 LUT 纹理。
            if (lutTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(lutTextureId), 0)
                lutTextureId = 0
            }
            loadedLutStyle = requestedStyle
            return
        }

        var bitmap: Bitmap? = null
        try {
            bitmap = context.resources.openRawResource(resId).use { input ->
                BitmapFactory.decodeStream(input)
            }
        } catch (e: Exception) {
            Log.w(TAG, "读取 LUT 资源失败 resId=$resId: ${e.message}", e)
        }
        if (bitmap == null) {
            Log.w(TAG, "LUT 解码失败（resId=$resId），该风格按原图渲染")
            if (lutTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(lutTextureId), 0)
                lutTextureId = 0
            }
            loadedLutStyle = requestedStyle
            return
        }

        try {
            if (bitmap.width != LUT_TEX_WIDTH || bitmap.height != LUT_TEX_HEIGHT) {
                Log.w(
                    TAG,
                    "LUT 尺寸异常: ${bitmap.width}x${bitmap.height}，" +
                        "期望 ${LUT_TEX_WIDTH}x${LUT_TEX_HEIGHT}",
                )
            }
            if (lutTextureId == 0) {
                val ids = IntArray(1)
                GLES20.glGenTextures(1, ids, 0)
                checkGl("glGenTextures(lut)")
                lutTextureId = ids[0]
            }
            GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, lutTextureId)
            // 手写三线性插值取的是纹素中心，必须 NEAREST，否则会跨 slice 混色。
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)
            checkGl("上传 LUT")
            loadedLutStyle = requestedStyle
            lutFailCount = 0
            Log.d(TAG, "LUT 已载入 style=$requestedStyle size=${bitmap.width}x${bitmap.height}")
        } catch (e: Exception) {
            lutFailCount++
            Log.w(TAG, "上传 LUT 失败（style=$requestedStyle，第 $lutFailCount 次），该风格按原图渲染: ${e.message}", e)
            if (lutTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(lutTextureId), 0)
                lutTextureId = 0
            }
            // 连续失败才放弃；否则下一帧再试一次（清过残留错误后通常一次就成）
            if (lutFailCount >= 3) loadedLutStyle = requestedStyle
        } finally {
            bitmap.recycle()
        }
    }

    // ==================================================================
    // 工具
    // ==================================================================

    private val executor: Executor = Executor { command -> handler.post(command) }

    private fun flushPendingClose() {
        if (pendingClose.isEmpty()) return
        // 下一帧再 close：此刻 EGLSurface 已经指向新的输出，旧 Surface 可以安全释放。
        for (o in pendingClose) {
            try {
                o.close()
            } catch (e: Exception) {
                Log.w(TAG, "关闭输出 Surface 出错: ${e.message}", e)
            }
        }
        pendingClose.clear()
    }

    private fun checkGl(op: String) {
        val error = GLES20.glGetError()
        if (error != GLES20.GL_NO_ERROR) {
            throw IllegalStateException("$op: GL error 0x" + Integer.toHexString(error))
        }
    }

    private class OutputPair(val surfaceOutput: SurfaceOutput, val surface: Surface)

    private class OutputEgl(
        val surface: Surface,
        val eglSurface: EGLSurface,
        val width: Int,
        val height: Int,
    )

    companion object {
        private const val TAG = "LutEffect"

        /**
         * 源像素是否走 sRGB → 线性。
         *
         * ★ 2026-10-04 定稿为 **false**：用 numpy 量了 res/raw 里这 7 张 LUT PNG 的中性轴
         *   （v=8→78、v=16→179、v=24→235，黑 0 / 白 255），它们在「sRGB 输入 → sRGB 输出」
         *   假设下几乎是恒等（最大偏差 46–53/255）；若按线性假设（true）喂进去，颜色平均
         *   偏 43/255、p95 偏 120/255 —— 差距肉眼可见。所以这批表要按 sRGB 直接查。
         *   将来换一批「线性输入」的表，把这个开关翻回 true 即可。
         */
        private const val LINEARIZE_SRGB = false

        /** LUT PNG 的固定尺寸：1024x32（32 个 slice，每 slice 32x32）。 */
        private const val LUT_TEX_WIDTH = 1024
        private const val LUT_TEX_HEIGHT = 32

        private const val TEX_SIZE = "uTexSize"
    }
}

/**
 * 一个 GLES2 program：编译/链接 + 属性与 uniform 位置缓存 + 绘制。
 *
 * 顶点数据是固定的一块全屏四边形（triangle strip，4 个顶点），所以直接把顶点数组
 * 交给 `glVertexAttribPointer`（客户端数组），不建 VBO。
 */
private class GlProgram(
    vertexShaderSource: String,
    fragmentShaderSource: String,
    private val texSizeUniform: String,
) {

    private val programHandle: Int
    private val positionLoc: Int
    private val texCoordLoc: Int
    private val inputSamplerLoc: Int
    private val lutSamplerLoc: Int
    private val matrixLoc: Int
    private val texSizeLoc: Int
    private val srgbLoc: Int
    private val scaleLoc: Int

    private val vertexBuffer: FloatBuffer = createFloatBuffer(VERTEX_COORDS)
    private val texCoordBuffer: FloatBuffer = createFloatBuffer(TEX_COORDS)

    init {
        val vertex = compile(GLES20.GL_VERTEX_SHADER, vertexShaderSource)
        val fragment = compile(GLES20.GL_FRAGMENT_SHADER, fragmentShaderSource)
        val handle = GLES20.glCreateProgram()
        checkGl("glCreateProgram")
        GLES20.glAttachShader(handle, vertex)
        GLES20.glAttachShader(handle, fragment)
        GLES20.glLinkProgram(handle)
        val linked = IntArray(1)
        GLES20.glGetProgramiv(handle, GLES20.GL_LINK_STATUS, linked, 0)
        if (linked[0] != GLES20.GL_TRUE) {
            val log = GLES20.glGetProgramInfoLog(handle)
            GLES20.glDeleteProgram(handle)
            throw IllegalStateException("链接 program 失败: $log")
        }
        GLES20.glDeleteShader(vertex)
        GLES20.glDeleteShader(fragment)
        programHandle = handle

        positionLoc = GLES20.glGetAttribLocation(handle, "aPosition")
        texCoordLoc = GLES20.glGetAttribLocation(handle, "aTextureCoord")
        inputSamplerLoc = GLES20.glGetUniformLocation(handle, "samplerInputTexture")
        lutSamplerLoc = GLES20.glGetUniformLocation(handle, "samplerLut")
        matrixLoc = GLES20.glGetUniformLocation(handle, "uTexMatrix")
        texSizeLoc = GLES20.glGetUniformLocation(handle, texSizeUniform)
        srgbLoc = GLES20.glGetUniformLocation(handle, "uSrgbToLinear")
        scaleLoc = GLES20.glGetUniformLocation(handle, "uLutScale")

        // 采样器固定绑到 0/1 号纹理单元，必须当前 program 处于 use 状态时才生效。
        GLES20.glUseProgram(handle)
        if (inputSamplerLoc >= 0) GLES20.glUniform1i(inputSamplerLoc, 0)
        if (lutSamplerLoc >= 0) GLES20.glUniform1i(lutSamplerLoc, 1)
        GLES20.glUseProgram(0)
    }

    fun use() {
        GLES20.glUseProgram(programHandle)
        GLES20.glEnableVertexAttribArray(positionLoc)
        GLES20.glVertexAttribPointer(positionLoc, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)
        GLES20.glEnableVertexAttribArray(texCoordLoc)
        GLES20.glVertexAttribPointer(texCoordLoc, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer)
    }

    fun setMatrix(matrix: FloatArray) {
        GLES20.glUniformMatrix4fv(matrixLoc, 1, false, matrix, 0)
    }

    /**
     * 设置「一个纹素的归一化尺寸」(1/宽, 1/高)。
     *
     * ★ 2026-10-04 踩坑：`glUniform*` **只作用于当前绑定的 program**。原来这里直接调
     *   `glUniform2f`，而调用点是在两个 program 都构造完、当前 program 已经被 init 里的
     *   `glUseProgram(0)` 解绑之后 → uniform 根本没写进去，`uTexSize` 一直是默认的 (0,0)
     *   → 着色器里 8 次采样坐标全变成 (0,0)、全落在 LUT 的黑角上 → 查表风格整体**全黑**。
     *   （用「固定采样 vec2(0.5,0.5)」的诊断着色器时画面是正常颜色，就是因为那条路径
     *   不经过 uTexSize。）所以这里必须先 use 本 program 再写 uniform。
     */
    fun setTexSize(width: Int, height: Int) {
        GLES20.glUseProgram(programHandle)
        GLES20.glUniform2f(texSizeLoc, 1f / width, 1f / height)
        GLES20.glUseProgram(0)
    }

    fun bindInputTexture(textureId: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
    }

    fun bindLutTexture(textureId: Int) {
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId)
    }

    fun setScale(scale: Float) {
        GLES20.glUniform1f(scaleLoc, scale)
    }

    fun setSrgbToLinear(enabled: Boolean) {
        GLES20.glUniform1f(srgbLoc, if (enabled) 1f else 0f)
    }

    fun draw() {
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, VERTEX_COUNT)
    }

    fun release() {
        GLES20.glDeleteProgram(programHandle)
    }

    private fun compile(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        checkGl("glCreateShader")
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val compiled = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0)
        if (compiled[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw IllegalStateException("编译 shader 失败(type=$type): $log")
        }
        return shader
    }

    private fun checkGl(op: String) {
        val error = GLES20.glGetError()
        if (error != GLES20.GL_NO_ERROR) {
            throw IllegalStateException("$op: GL error 0x" + Integer.toHexString(error))
        }
    }

    companion object {
        private const val VERTEX_COUNT = 4

        private val VERTEX_COORDS = floatArrayOf(
            -1f, -1f,
            1f, -1f,
            -1f, 1f,
            1f, 1f,
        )

        private val TEX_COORDS = floatArrayOf(
            0f, 0f,
            1f, 0f,
            0f, 1f,
            1f, 1f,
        )

        fun createFloatBuffer(coords: FloatArray): FloatBuffer {
            val bb = ByteBuffer.allocateDirect(coords.size * 4)
            bb.order(ByteOrder.nativeOrder())
            val fb = bb.asFloatBuffer()
            fb.put(coords)
            fb.position(0)
            return fb
        }
    }
}

// ==================================================================
// Shader 源码
// ==================================================================

private const val LUT_VERTEX_SHADER = """
attribute vec4 aPosition;
attribute vec4 aTextureCoord;
uniform mat4 uTexMatrix;
varying vec2 vTextureCoord;
void main() {
    gl_Position = aPosition;
    vTextureCoord = (uTexMatrix * aTextureCoord).xy;
}
"""

/**
 * 3D LUT 采样。
 *
 * 纹理是 1024x32 的横向切片：`x = b*32 + r`、`y = g`，r/g/b 都是 0..31 的输入索引
 * （所以纹理列方向红最快、行方向绿、slice 方向蓝）。NEAREST 采样 + 手写三线性插值：
 *
 * - 先由归一化坐标算出 0..31 的连续索引；
 * - 底层 = floor、顶层 = min(floor+1, 31)（越界就夹住，等于用边界值外推）；
 * - 在 r 与 g 上各做一次双线性得到两个值，再按 b 的小数部分插值。
 *
 * `uLutScale` 为 1 时走查表，为 0 时（原图 / LUT 载入失败）直接输出输入色。
 * 两种风格共用 `uSrgbToLinear`：为 1 时先把输入从 sRGB 转到线性再查表，
 * 保证色调映射在线性空间完成；要退回「直接查表」只需把这个开关置 0。
 */
private const val LUT_FRAGMENT_SHADER = """
#extension GL_OES_EGL_image_external : require
precision highp float;
varying vec2 vTextureCoord;
uniform samplerExternalOES samplerInputTexture;
uniform sampler2D samplerLut;
uniform vec2 uTexSize;
uniform float uLutScale;
uniform float uSrgbToLinear;

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 sampleLut(vec3 c) {
    float last = 31.0;
    // ★ 这里的 c 是**归一化到 0..1** 的通道（不是 0..255！），所以直接乘 31：
    //   c=0 → 第 0 档、c=1 → 第 31 档，正好落在表的两端。
    //   踩过的坑：照抄 CPU 侧 8bit 的式子写成 c * (31.0 / 255.0)，归一化输入下
    //   索引只有 0..0.12 → floor 恒为 0 → 每个像素都去查表的黑色角 → 整屏全黑。
    //   （CPU 侧 `LutPostProcess` 的输入是 0..255，那边用 31/255 才是对的。）
    vec3 p = c * last;

    float r0 = floor(p.r);
    float g0 = floor(p.g);
    float b0 = floor(p.b);
    float r1 = min(r0 + 1.0, last);
    float g1 = min(g0 + 1.0, last);
    float b1 = min(b0 + 1.0, last);
    float fr = p.r - r0;
    float fg = p.g - g0;
    float fb = p.b - b0;

    // 纹理是 1024x32：列方向 x = b*32 + r（红最快），行方向 y = g（绿），slice 方向蓝。
    // 采到纹素中心，所以 +0.5 后再除以纹理尺寸。
    float y0 = (g0 + 0.5) * uTexSize.y;
    float y1 = (g1 + 0.5) * uTexSize.y;
    float x00 = (b0 * 32.0 + r0 + 0.5) * uTexSize.x;
    float x10 = (b0 * 32.0 + r1 + 0.5) * uTexSize.x;
    float x01 = (b1 * 32.0 + r0 + 0.5) * uTexSize.x;
    float x11 = (b1 * 32.0 + r1 + 0.5) * uTexSize.x;

    vec3 c00 = texture2D(samplerLut, vec2(x00, y0)).rgb;
    vec3 c10 = texture2D(samplerLut, vec2(x10, y0)).rgb;
    vec3 c01 = texture2D(samplerLut, vec2(x01, y0)).rgb;
    vec3 c11 = texture2D(samplerLut, vec2(x11, y0)).rgb;
    vec3 c02 = texture2D(samplerLut, vec2(x00, y1)).rgb;
    vec3 c12 = texture2D(samplerLut, vec2(x10, y1)).rgb;
    vec3 c03 = texture2D(samplerLut, vec2(x01, y1)).rgb;
    vec3 c13 = texture2D(samplerLut, vec2(x11, y1)).rgb;

    // 先在 r、g 上各做双线性（每个蓝 slice 一项），再按 b 的小数部分插值 —— 共 8 次采样。
    vec3 blue0 = mix(mix(c00, c10, fr), mix(c02, c12, fr), fg);
    vec3 blue1 = mix(mix(c01, c11, fr), mix(c03, c13, fr), fg);
    return mix(blue0, blue1, fb);
}

void main() {
    vec3 src = texture2D(samplerInputTexture, vTextureCoord).rgb;
    // 查表前先按开关做 sRGB -> 线性（LUT 的输入轴是线性光强）。
    if (uSrgbToLinear > 0.5) {
        src = srgbToLinear(src);
    }
    // LUT 里存的就是最终 sRGB 显示值：查表结果直接输出，不做二次变换。
    // uLutScale 为 0（原图 / LUT 载入失败）时退回输入色，保证降级是「原图」而不是黑屏。
    gl_FragColor = vec4(mix(src, sampleLut(src), uLutScale), 1.0);
}
"""

/** 黑白：Rec.709 亮度 + 轻微提对比，不查表。 */
private const val BW_FRAGMENT_SHADER = """
#extension GL_OES_EGL_image_external : require
precision highp float;
varying vec2 vTextureCoord;
uniform samplerExternalOES samplerInputTexture;
uniform float uSrgbToLinear;

vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

void main() {
    vec3 src = texture2D(samplerInputTexture, vTextureCoord).rgb;
    // sRGB(Rec.709) 感知亮度做灰度；uSrgbToLinear 控制是否再把结果转成线性输出，
    // 与查表风格共用同一个开关，保证两种风格输出色彩空间一致。
    float y = dot(src, vec3(0.2126, 0.7152, 0.0722));
    // 轻微 S 曲线提对比，围着中灰 0.5 收一下。
    y = clamp((y - 0.5) * 1.10 + 0.5, 0.0, 1.0);
    vec3 color = vec3(y);
    if (uSrgbToLinear > 0.5) {
        color = srgbToLinear(color);
    }
    gl_FragColor = vec4(color, 1.0);
}
"""
