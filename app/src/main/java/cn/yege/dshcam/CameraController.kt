package cn.yege.dshcam

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Rect
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.util.Log
import android.util.Size
import android.view.Surface
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: androidx.camera.view.PreviewView,
    private val analyzerFactory: () -> ImageAnalysis.Analyzer,
    private val onError: (String) -> Unit,
    private val onSaved: () -> Unit = {}
) {
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var preview: Preview? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var imageCapture: ImageCapture? = null
    private val mainExecutor: Executor by lazy { ContextCompat.getMainExecutor(context) }
    private var ioExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // ★ 2026-10-04 新增：实拍参数控制（点屏对焦 / 双指缩放 / 闪光灯 / 曝光补偿）
    private var cameraControl: CameraControl? = null
    private var cameraInfo: CameraInfo? = null
    private var flashMode: Int = ImageCapture.FLASH_MODE_OFF

    // ★ 2026-10-04 新增：画幅。CameraX 1.4.1 的 AspectRatio 只有 RATIO_4_3 / RATIO_16_9
    //   （javap 查过 camera-core-1.4.1.aar，没有 RATIO_1_1），所以：
    //   · 4:3 / 16:9 交给 CameraX 按宽高比挑分辨率（16:9 是原生裁剪，画质不降）
    //   · 1:1 预览与拍摄都按 4:3 跑，拍完在 cropToSquare() 里居中裁成正方形
    private var ratioMode: Int = RATIO_43

    // ★ 2026-10-04：实时调色（LUT）。GL 线程/processor/effect 只建一次，
    //   切风格只改 @Volatile 的序号，**不重绑 use case**（重绑会黑屏闪一下）
    private var glExecutor: ExecutorService? = null
    private var lutEffect: LutEffect? = null
    private var lutStyle: Int = LutStyles.ORIGINAL

    /** 重建会话时要补回的变焦倍率 / 曝光档位（重绑后 CameraX 会复位） */
    private var lastZoomRatio = 1f
    private var lastEvIndex = 0

    // ★ 2026-10-05 新增（自动构图 B）：开着的时候，拍完除了原图，再另存一张
    //   「自动构图版」（把主体挪到三分点、顺手拉直画面）到同目录，文件名加 _auto 后缀。
    private var autoFrameEnabled = true

    /** 自动构图开关（第 7 颗胶囊控制） */
    fun setAutoFrameEnabled(on: Boolean) {
        autoFrameEnabled = on
    }

    fun autoFrameEnabled(): Boolean = autoFrameEnabled

    // ★ 仅调试用（`--ez forcetrim true`）：跳过"构图已达标就不另存"的判断，
    //   强制裁一刀，用来验证另存链路本身（写 MediaStore + 写回 EXIF 方向 + 相册扫描）。
    private var debugForceTrim = false

    fun setDebugForceTrim(on: Boolean) {
        debugForceTrim = on
    }

    companion object {
        private const val TAG = "CameraController"
        const val RATIO_43 = 0
        const val RATIO_169 = 1
        const val RATIO_SQUARE = 2
    }

    /** 切换画幅；相机已启动的话立刻重建用例（预览马上变） */
    fun setCaptureRatio(mode: Int) {
        if (mode < RATIO_43 || mode > RATIO_SQUARE || mode == ratioMode) return
        ratioMode = mode
        if (cameraProvider != null) bindCamera()
    }

    fun captureRatio(): Int = ratioMode

    // 1:1 没有原生宽高比，预览/拍摄都退回 4:3（拍完再裁）
    private fun previewAspectRatio(): Int =
        if (ratioMode == RATIO_169) AspectRatio.RATIO_16_9 else AspectRatio.RATIO_4_3

    /**
     * 成片用的分辨率选择器：优先 4096 宽那一档（4:3 → 4096×3072，16:9 → 4096×2304），
     * 该机型没这一档就往下降一档找。目的：别让实时调色的中间 Surface 把成片压成 1080p。
     */
    private fun captureResolutionSelector(): ResolutionSelector {
        val target = if (ratioMode == RATIO_169) Size(4096, 2304) else Size(4096, 3072)
        val strategy = if (ratioMode == RATIO_169) {
            AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
        } else {
            AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY
        }
        return ResolutionSelector.Builder()
            .setAspectRatioStrategy(strategy)
            .setResolutionStrategy(
                ResolutionStrategy(target, ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
            )
            .build()
    }

    fun start() {
        try {
            if (ioExecutor.isShutdown) {
                ioExecutor = Executors.newSingleThreadExecutor()
            }
            // ★ 实时调色：GL 执行器 + CameraEffect（只建一次，切风格不重绑）
            if (glExecutor == null || glExecutor?.isShutdown == true) {
                glExecutor = Executors.newSingleThreadExecutor()
            }
            if (lutEffect == null) {
                val executor = glExecutor
                if (executor != null) {
                    // 注意：必须走 LutEffect.create —— 处理器实例要在 super(...) 之前造出来，
                    // 所以构造函数是 private 的，外面拿不到
                    lutEffect = LutEffect.create(context, executor) { t ->
                        Log.w(TAG, "LUT 调色出错: ${t.localizedMessage}")
                    }
                    lutEffect?.setStyle(lutStyle)
                }
            }
            val providerFuture = ProcessCameraProvider.getInstance(context)
            providerFuture.addListener({
                try {
                    cameraProvider = providerFuture.get()
                    bindCamera()
                } catch (e: Exception) {
                    onError("无法获取相机服务: ${e.localizedMessage}")
                }
            }, mainExecutor)
        } catch (e: Exception) {
            onError("启动相机失败: ${e.localizedMessage}")
        }
    }

    fun stop() {
        try {
            cameraProvider?.unbindAll()
            cameraProvider = null
            preview = null
            imageAnalysis = null
            imageCapture = null
            cameraControl = null
            cameraInfo = null
            // 实时调色的 GL 资源要在解绑之后放
            lutEffect?.release()
            lutEffect = null
            glExecutor?.shutdown()
            glExecutor = null
            ioExecutor.shutdown()
        } catch (e: Exception) {
            onError("停止相机失败: ${e.localizedMessage}")
        }
    }

    fun switchLens() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        bindCamera()
    }

    fun capture() {
        try {
            val capture = imageCapture ?: run {
                onError("相机未初始化")
                return
            }

            val timeFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            val fileName = "DSHCam_${timeFormat.format(Date())}.jpg"
            val contentValues = ContentValues().apply {
                put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/DSHCam")
                    put(android.provider.MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val outputOptions = ImageCapture.OutputFileOptions.Builder(
                context.contentResolver,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            ).build()

            // ★ 必须用「直接写文件」的 OnImageSavedCallback。
            //   原来用 OnImageCapturedCallback 把内存里的 ImageProxy 按 NV21 三平面转换，
            //   但 in-flight 捕获回调返回的是 **JPEG 格式（只有 1 个 plane）** → planes[1] 越界
            //   → 转换函数 catch 住返回 null → 弹「图片转换失败」，一张都存不下来
            //   （2026-10-04 在小米 14 Ultra 上实机复现）
            capture.takePicture(
                outputOptions,
                mainExecutor,
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        val uri = output.savedUri
                        if (uri == null) {
                            onSaved()
                            return
                        }
                        // 拍后处理两件事：
                        //   · 1:1 画幅要补一刀居中裁剪（CameraX 没有正方形的原生宽高比）
                        //   · 选了风格要用 CPU 版 LUT 给成片补调色 —— 预览那条 CameraEffect
                        //     只挂 PREVIEW，因为把 IMAGE_CAPTURE 交给它以后成片分辨率会掉到
                        //     效果输入 Surface 的尺寸（实测 12MP → 1080×1920，显式
                        //     ResolutionSelector 也拉不回来）
                        val needCrop = ratioMode == RATIO_SQUARE
                        val needGrade = lutStyleIndex() != LutStyles.ORIGINAL
                        val needAutoFrame = autoFrameEnabled
                        val needPost = needCrop || needGrade || needAutoFrame
                        val finish = Runnable {
                            try {
                                if (needCrop) cropToSquare(uri)
                                if (needGrade) gradeInPlace(uri)
                                // ★ 自动构图版放在最后：从已经裁好/调好的成片再派生一张
                                if (needAutoFrame) writeAutoFrame(uri)
                                // 清掉 IS_PENDING，图库才会显示这张图
                                val done = ContentValues().apply {
                                    put(android.provider.MediaStore.Images.Media.IS_PENDING, 0)
                                }
                                context.contentResolver.update(uri, done, null, null)
                                MediaScannerConnection.scanFile(
                                    context,
                                    arrayOf(uri.toString()),
                                    arrayOf("image/jpeg"),
                                    null
                                )
                            } catch (e: Exception) {
                                val msg = "保存后处理失败: ${e.localizedMessage}"
                                mainExecutor.execute { onError(msg) }
                            }
                            mainExecutor.execute { onSaved() }
                        }
                        if (needPost) {
                            // 裁剪 / 调色都要解图 + 重编码（12MP 约 0.3~2 秒），放后台线程，别卡 UI
                            try {
                                if (ioExecutor.isShutdown) {
                                    ioExecutor = Executors.newSingleThreadExecutor()
                                }
                                ioExecutor.execute(finish)
                            } catch (e: Exception) {
                                finish.run()
                            }
                        } else {
                            finish.run()
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        onError("拍照失败: ${exception.localizedMessage}")
                    }
                }
            )
        } catch (e: Exception) {
            onError("拍照失败: ${e.localizedMessage}")
        }
    }

    // ------------------------------------------------------------------
    // ★ 2026-10-04 新增：实拍参数控制（用户要求"都加"）
    //   这些开关都作用在"成片"上，不再是屏幕提示。
    // ------------------------------------------------------------------

    /**
     * 点屏幕对焦 + 测光。xNorm / yNorm 是相对 PreviewView 的归一化坐标（0..1）。
     * 对应 CameraX 的 FocusMeteringAction：AF（对焦）+ AE（测光）一起做。
     */
    fun focusAt(xNorm: Float, yNorm: Float) {
        val control = cameraControl ?: return
        try {
            val w = previewView.width
            val h = previewView.height
            if (w <= 0 || h <= 0) return
            val point = previewView.meteringPointFactory.createPoint(xNorm * w, yNorm * h)
            val action = FocusMeteringAction
                .Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(3, TimeUnit.SECONDS)
                .build()
            control.startFocusAndMetering(action)
        } catch (e: Exception) {
            onError("对焦失败: ${e.localizedMessage}")
        }
    }

    /** 双指缩放：scaleFactor 是相对倍率（>1 放大）。范围按机型的 zoomState 夹紧。 */
    fun zoomBy(scaleFactor: Float) {
        val control = cameraControl ?: return
        val state = cameraInfo?.zoomState?.value ?: return
        try {
            val target = (state.zoomRatio * scaleFactor)
                .coerceIn(state.minZoomRatio, state.maxZoomRatio)
            control.setZoomRatio(target)
            lastZoomRatio = target
        } catch (e: Exception) {
            // 超范围/相机正在切换时会抛，忽略即可
        }
    }

    /** 当前变焦倍率（UI 显示用） */
    fun currentZoom(): Float = cameraInfo?.zoomState?.value?.zoomRatio ?: 1f

    /**
     * 切换实时调色风格（0=原图 … 8=黑白）。
     *
     * ★ 2026-10-04 真机实测：已经跑起来的画面里，让**同一个效果实例**上传一张新的
     *   LUT 贴图，会把预览刷成全黑（截图取景区亮度 1.8，之后一直黑；同风格重复设置、
     *   或切到不需要贴图的「黑白」都正常；启动时用 `--ei style N` 预置也正常）。
     *   → 换风格时不要复用旧实例，直接把效果**整个重建**（新的 GL 上下文/纹理/程序，
     *   上传发生在它自己的 onInputSurface 初始化路径里，等于「启动时预置」那条路），
     *   再重绑一次 use case；代价是换风格时预览闪一下，变焦/曝光由 bindCamera() 补回。
     */
    fun setLutStyle(index: Int) {
        val safe = if (index in 0 until LutStyles.names.size) index else LutStyles.ORIGINAL
        if (safe == lutStyle) return
        lutStyle = safe
        rebuildLutEffect()
        if (cameraProvider != null) bindCamera()
    }

    /** 重建调色效果实例（换风格时用，绕开「运行中上传 LUT」那个坑）。 */
    private fun rebuildLutEffect() {
        try {
            lutEffect?.release()
        } catch (e: Exception) {
            Log.w(TAG, "释放旧调色效果失败: ${e.localizedMessage}")
        }
        lutEffect = null
        val exec = glExecutor
        if (exec == null) return
        try {
            lutEffect = LutEffect.create(context, exec) { t ->
                Log.w(TAG, "LUT 调色出错: ${t.localizedMessage}")
            }
            lutEffect?.setStyle(lutStyle)
        } catch (e: Exception) {
            Log.w(TAG, "创建调色效果失败: ${e.localizedMessage}")
        }
    }

    fun lutStyleIndex(): Int = lutStyle

    /** 闪光灯切档：关 → 自动 → 开 → 关；返回新档位（ImageCapture.FLASH_MODE_*） */
    fun cycleFlash(): Int {
        flashMode = when (flashMode) {
            ImageCapture.FLASH_MODE_OFF -> ImageCapture.FLASH_MODE_AUTO
            ImageCapture.FLASH_MODE_AUTO -> ImageCapture.FLASH_MODE_ON
            else -> ImageCapture.FLASH_MODE_OFF
        }
        // 相机已 bind 就直接生效；没 bind 时先记着，bindCamera() 里会补上
        imageCapture?.flashMode = flashMode
        return flashMode
    }

    /** 曝光补偿可选档位（android.util.Range，通常 -12..12 档） */
    fun exposureRange(): android.util.Range<Int>? =
        cameraInfo?.exposureState?.exposureCompensationRange

    /** 一档 index 等于多少 EV（通常 1/3 EV） */
    fun exposureStep(): Float =
        cameraInfo?.exposureState?.exposureCompensationStep?.toFloat() ?: (1f / 3f)

    /** 曝光补偿：index 是档位，会按机型范围夹紧 */
    fun setExposureIndex(index: Int) {
        val control = cameraControl ?: return
        val range = exposureRange() ?: return
        try {
            val clamped = index.coerceIn(range.lower, range.upper)
            control.setExposureCompensationIndex(clamped)
            lastEvIndex = clamped
        } catch (e: Exception) {
            // 忽略
        }
    }

    /**
     * 给刚存下的成片补上「风格」调色：用与预览 shader 逐行等价的 CPU 版本
     * [LutPostProcess] 原地改写同一个 MediaStore 文件。
     *
     * 为什么不在预览那条效果链上顺手做：把 IMAGE_CAPTURE 交给 CameraEffect 之后，
     * 成片分辨率会掉到效果输入 Surface 的尺寸（实测 12MP → 1080×1920，显式
     * ResolutionSelector 也拉不回来）。所以效果只挂 PREVIEW，成片在这里补。
     *
     * 注意：JPEG 解出来的是**传感器方向**的像素（旋转记在 EXIF 里），重编码不会带
     * EXIF，所以要先读原图 Orientation、写完再写回，否则图库里照片会躺着。
     * 任何一步失败都只 log 不动原图 —— 宁可给一张没调色的，也不能把照片弄坏。
     */
    private fun gradeInPlace(uri: Uri) {
        val style = lutStyleIndex()
        if (style == LutStyles.ORIGINAL) return
        val resolver = context.contentResolver

        // 1) 先记下原图的 EXIF 方向（重编码会丢 EXIF）
        var orientation: String? = null
        try {
            resolver.openInputStream(uri)?.use { ins ->
                orientation = ExifInterface(ins).getAttribute(ExifInterface.TAG_ORIENTATION)
            }
        } catch (e: Exception) {
            Log.w(TAG, "调色：读 EXIF 失败 ${e.localizedMessage}")
        }

        // 2) 解图 → CPU 调色 → 覆盖写回
        val decoded = try {
            resolver.openInputStream(uri)?.use { ins -> BitmapFactory.decodeStream(ins) }
        } catch (e: Exception) {
            Log.w(TAG, "调色：解图异常 ${e.localizedMessage}")
            null
        }
        if (decoded == null) {
            Log.w(TAG, "调色跳过：解不开刚存的成片")
            return
        }
        var output: Bitmap? = null
        try {
            val w = decoded.width
            val h = decoded.height
            if (w <= 0 || h <= 0) return
            val pixels = IntArray(w * h)
            decoded.getPixels(pixels, 0, w, 0, 0, w, h)
            decoded.recycle()   // 先放掉 48MB 的输入位图，给像素数组和目标位图腾地方
            LutPostProcess.apply(context, pixels, w, h, style)
            // ★ BitmapFactory 解出来的是**不可变**位图，直接 setPixels 会抛
            //   java.lang.IllegalStateException: null（实测 at android.graphics.Bitmap.setPixels）
            //   → 必须另建一张可写位图来装调色结果
            val graded = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            output = graded
            graded.setPixels(pixels, 0, w, 0, 0, w, h)
            resolver.openOutputStream(uri, "wt")?.use { os ->
                graded.compress(Bitmap.CompressFormat.JPEG, 95, os)
            } ?: Log.w(TAG, "调色跳过：打不开输出流")
        } catch (e: Exception) {
            Log.w(TAG, "调色失败（保留原图）: ${e.javaClass.name}: ${e.message}", e)
            return
        } finally {
            output?.recycle()
        }

        // 3) 把方向写回去
        if (orientation != null) {
            try {
                resolver.openFileDescriptor(uri, "rw")?.use { pfd ->
                    val exif = ExifInterface(pfd.fileDescriptor)
                    exif.setAttribute(ExifInterface.TAG_ORIENTATION, orientation)
                    exif.saveAttributes()
                }
            } catch (e: Exception) {
                Log.w(TAG, "调色：写回 EXIF 方向失败 ${e.localizedMessage}")
            }
        }
    }

    /**
     * ★ 2026-10-05 新增（自动构图 B）：另存一张「自动构图版」。
     *
     * 流程：读刚保存的成片 → 缩到 64×48 算显著性图（边缘 0.6 + 肤色 0.25 + 饱和 0.15，
     * 算法在 `AutoFrame.kt`，纯 Kotlin 有单测）→ `AutoFrame.bestCrop` 找最佳裁剪框
     * （打分 = 显著性覆盖率 0.45 + 三分点 0.30 + 面积 0.25 − 切掉显著区 2.0）→
     * 按框裁一张新的 JPEG，名字加 `_auto` 后缀，写进 MediaStore 同一个相册目录。
     *
     * 保守原则（和 cropToSquare 一样）：任何一步失败都只写 Log，绝不动原图；
     * 构图本来就达标（裁剪框几乎等于整图）时不另存，免得存一张一模一样的重复图。
     */
    private fun writeAutoFrame(uri: Uri) {
        val resolver = context.contentResolver
        val srcName = displayNameOf(uri) ?: return

        // 1) 先记 EXIF 方向（重编码会丢，得给新文件写上同一个方向）
        var orientation = ExifInterface.ORIENTATION_NORMAL
        try {
            resolver.openInputStream(uri)?.use { ins ->
                orientation = ExifInterface(ins).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "自动构图：读 EXIF 失败 ${e.localizedMessage}")
        }

        // 2) 解图（BitmapFactory 不会帮我们转向，所以这里的坐标系是"传感器方向"；
        //    好在 90° 旋转下三分点依然是三分点，裁剪框按同一宽高比求，结果是对的）
        val src = try {
            resolver.openInputStream(uri)?.use { ins -> BitmapFactory.decodeStream(ins) }
        } catch (e: Exception) {
            Log.w(TAG, "自动构图：解图异常 ${e.localizedMessage}")
            null
        }
        if (src == null) {
            Log.w(TAG, "自动构图跳过：解不开成片")
            return
        }

        var small: Bitmap? = null
        var cut: Bitmap? = null
        try {
            val w = src.width
            val h = src.height
            if (w < 256 || h < 256) return

            // 3) 显著性图（64×48 网格）。规则版先算好：既是模型不可用时的兜底，
            //    也是日志里"模型 vs 规则"的对照。
            val gw = 64
            val gh = 48
            small = Bitmap.createScaledBitmap(src, gw, gh, true)
            val argb = IntArray(gw * gh)
            small.getPixels(argb, 0, gw, 0, 0, gw, gh)

            // ★ 2026-10-05 起优先用主体模型（u2netp，assets 里 4.4 MB），
            //   模型不可用（加载/推理失败）才回退到规则算法。模型在「用户看到的方向」之前
            //   的坐标系上算 —— 和下面一样是传感器方向，后面的转正/映射逻辑完全不变。
            var sensorSalience = AutoFrame.salienceFromArgb(argb, gw, gh, gw, gh)
            var modelPeak = -1f
            if (SubjectModel.ensureLoaded(context)) {
                val grid = SubjectModel.salienceGrid(src, gw, gh)
                if (grid != null) {
                    sensorSalience = AutoFrame.Salience(gw, gh, grid)
                    modelPeak = SubjectModel.lastPeak
                    Log.d(TAG, "自动构图：主体模型 ${SubjectModel.lastInfo}")
                }
            }
            // 3a) 模型自己承认"画面里没有主体"（离线实测：空场景峰值 0.573，有主体 0.996）
            //     → 直接不另存。这条正是用户抱怨的「没主体也乱裁/一直催」的根治办法。
            if (modelPeak >= 0f && modelPeak < SaliencyMath.GRID_PEAK_GATE && !debugForceTrim) {
                Log.d(TAG, "自动构图：主体模型判定画面里没有明显主体，不另存（峰值 ${"%.3f".format(modelPeak)}）" +
                    "｜${SubjectModel.lastInfo}")
                return
            }

            // 3b) ★ 2026-10-05 修：构图必须在「用户看到的方向」上判。
            //     竖拍（EXIF 6/8）时传感器是横的、成片是竖的；直接在传感器坐标系里判，
            //     "主体偏在画面右边"可能正好落在横坐标系的三分点上 → 误判"已达标"、不裁
            //     （用户实拍的黑色鼠标照片就是这样）。先把显著性网格转正再判。
            val quarterTurn = orientation == ExifInterface.ORIENTATION_ROTATE_90 ||
                orientation == ExifInterface.ORIENTATION_ROTATE_270
            val shown = AutoFrame.rotateSalience(sensorSalience, orientation)
            val shownW = if (quarterTurn) h else w
            val shownH = if (quarterTurn) w else h

            // 4) 最佳裁剪框（保持原宽高比，最多裁掉 40% 面积）
            //    ★ 2026-10-05 修：原来 minKeep 给的是 0.85（只能裁 15% 面积），
            //    加上评分里"裁掉显著度"的罚分过重、面积项权重过大，实测两张真机照片
            //    都判"构图已达标"，自动构图从不生效。现在放宽到 0.60，让算法真的能重新构图。
            //    框在"显示方向"上求，再映射回传感器坐标去裁原图（面积比不变）。
            val cropShown = AutoFrame.bestCrop(
                shown, shownW, shownH, shownW.toFloat() / shownH.toFloat(), 0.60f
            )
            val crop = AutoFrame.mapCropToSensor(cropShown, w, h, orientation)
            val cw = crop.width
            val chh = crop.height
            if (cw <= 0 || chh <= 0) return
            val left = crop.left.coerceIn(0, w - 1)
            val top = crop.top.coerceIn(0, h - 1)
            val right = (left + cw).coerceAtMost(w)
            val bottom = (top + chh).coerceAtMost(h)
            val rw = right - left
            val rh = bottom - top
            if (rw < 64 || rh < 64) return
            // 只裁掉不到 3% 面积 → 等于没动，不另存重复图（原判据是"少 8 像素"，
            // 在高分辨率成片上永远成立，会把有效的小幅重组也一起挡掉）
            val keepArea = rw.toDouble() * rh.toDouble() / (w.toDouble() * h.toDouble())
            var outLeft = left
            var outTop = top
            var outW = rw
            var outH = rh
            if (keepArea >= 0.97) {
                if (!debugForceTrim) {
                    Log.d(TAG, "自动构图：构图已达标，不另存（保留 ${"%.1f".format(keepArea * 100)}%，分数 ${crop.score}）" +
                        "｜评分依据 " + AutoFrame.lastScoreInfo)
                    return
                }
                // 调试钩子（--ez forcetrim true）：算法认为当前画面不用裁，但为了验证
                // "另存自动构图版"这条链路（MediaStore 写入 + EXIF 方向 + 相册扫描），
                // 强制居中裁掉约 18% 面积，产出一张能看出差别的 _xxx_auto.jpg。
                val f = 0.905
                outW = (w * f).toInt().coerceAtLeast(64)
                outH = (h * f).toInt().coerceAtLeast(64)
                outLeft = (w - outW) / 2
                outTop = (h - outH) / 2
                Log.d(TAG, "自动构图：调试强制裁剪 ${w}×${h} → ${outW}×${outH}")
            }

            // 5) 裁 + 写新文件
            cut = Bitmap.createBitmap(src, outLeft, outTop, outW, outH)
            val newUri = saveAutoBitmap(cut, srcName, orientation)
            Log.d(TAG, "自动构图：已另存 $newUri（${w}×${h} → ${outW}×${outH} 分数 ${crop.score}）" +
                "｜评分依据 " + AutoFrame.lastScoreInfo)
        } catch (e: Exception) {
            Log.w(TAG, "自动构图失败（保留原图）: ${e.javaClass.name}: ${e.message}", e)
        } finally {
            try { small?.recycle() } catch (_: Exception) {}
            try { cut?.recycle() } catch (_: Exception) {}
            src.recycle()
        }
    }

    /** MediaStore 里那张图的显示文件名（含扩展名） */
    private fun displayNameOf(uri: Uri): String? = try {
        context.contentResolver.query(
            uri, arrayOf(android.provider.MediaStore.Images.Media.DISPLAY_NAME), null, null, null
        )?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
    } catch (e: Exception) {
        null
    }

    /**
     * 把裁好的位图作为「新照片」写进相机相册：名字 `<原名>_auto.jpg`，
     * 写的时候 IS_PENDING=1（图库先别显示半张），写完补上 EXIF 方向再置 0。
     */
    private fun saveAutoBitmap(bmp: Bitmap, srcName: String, orientation: Int): Uri? {
        val resolver = context.contentResolver
        val base = srcName.substringBeforeLast('.', srcName)
        val values = ContentValues().apply {
            put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, "${base}_auto.jpg")
            put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    android.provider.MediaStore.Images.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/DSHCam"
                )
                put(android.provider.MediaStore.Images.Media.IS_PENDING, 1)
            }
        }
        val newUri = resolver.insert(
            android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
        ) ?: return null
        resolver.openOutputStream(newUri)?.use { os ->
            bmp.compress(Bitmap.CompressFormat.JPEG, 95, os)
        }
        // EXIF 方向写回（从原图抄），否则 90° 拍的自动版会躺倒
        try {
            resolver.openFileDescriptor(newUri, "rw")?.use { pfd ->
                ExifInterface(pfd.fileDescriptor).apply {
                    setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString())
                    saveAttributes()
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "自动构图：写 EXIF 方向失败 ${e.localizedMessage}")
        }
        val done = ContentValues().apply {
            put(android.provider.MediaStore.Images.Media.IS_PENDING, 0)
        }
        resolver.update(newUri, done, null, null)
        MediaScannerConnection.scanFile(context, arrayOf(newUri.toString()), arrayOf("image/jpeg"), null)
        return newUri
    }

    /**
     * 1:1 画幅：CameraX 没有 RATIO_1_1（camera-core 1.4.1 里只有 4:3 / 16:9，javap 确认过），
     * 所以拍完把 JPEG 居中裁成正方形。
     *
     * 用 BitmapRegionDecoder 只解出中间那块（比整图解码省一半以上内存），先写到 cacheDir
     * 的临时文件（顺手把原图的 EXIF 方向写回去），再整段覆盖 MediaStore 里那张原图。
     * 正方形裁剪对 90° 旋转是对称的，所以不用管 orientation=6/8 的坐标系差异。
     * 任何一步失败都直接返回，保持原图不动 —— 宁可给用户一张 4:3，也不能把照片弄坏。
     */
    private fun cropToSquare(uri: Uri) {
        var tmp: File? = null
        try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, bounds)
            }
            val w = bounds.outWidth
            val h = bounds.outHeight
            if (w <= 0 || h <= 0) return
            val side = minOf(w, h)
            if (side == w && side == h) return          // 本来就是正方形
            val rect = Rect((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2)

            val orientation = context.contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL

            var region: Bitmap? = null
            @Suppress("DEPRECATION")
            context.contentResolver.openInputStream(uri)?.use { ins ->
                val decoder = BitmapRegionDecoder.newInstance(ins, false) ?: return
                region = decoder.decodeRegion(rect, BitmapFactory.Options())
                decoder.recycle()
            }
            val cropped = region ?: return

            tmp = File.createTempFile("dshcam_square", ".jpg", context.cacheDir)
            FileOutputStream(tmp).use { out ->
                cropped.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
            cropped.recycle()
            ExifInterface(tmp.absolutePath).apply {
                setAttribute(ExifInterface.TAG_ORIENTATION, orientation.toString())
                saveAttributes()
            }
            context.contentResolver.openOutputStream(uri, "wt")?.use { out ->
                tmp.inputStream().use { it.copyTo(out) }
            }
        } catch (e: Exception) {
            Log.w(TAG, "裁正方形失败，保留原图: ${e.localizedMessage}")
        } finally {
            tmp?.delete()
        }
    }

    private fun bindCamera() {
        try {
            cameraProvider?.unbindAll()

            preview = Preview.Builder()
                .setTargetRotation(previewView.display.rotation)
                .setTargetAspectRatio(previewAspectRatio())
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // 分析流必须跟预览同宽高比：原来写死 1280×720（16:9）而预览是 4:3，
            // 于是人脸框的归一化坐标跟预览对不上（实机截图里框会飘到画面角落）
            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetAspectRatio(previewAspectRatio())
                .build()
                .also { analysis ->
                    val analyzer = analyzerFactory()
                    analysis.setAnalyzer(ioExecutor, analyzer)
                }

            // ★ 成片分辨率：挂了 CameraEffect（实时调色）之后，ImageCapture 如果只给
            //   setTargetAspectRatio，CameraX 会按效果链的中间 Surface 尺寸出图，
            //   实测掉到 1080×1920（2 MP，从 4096×3072 掉下来）。这里显式指定
            //   「接近 4096 宽的最高分辨率 + 对应宽高比」，把成片拉回 12 MP。
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setResolutionSelector(captureResolutionSelector())
                .build()

            val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            // ★ 用 UseCaseGroup 把三个 use case 和实时调色效果绑在一起：
            //   addEffect 让预览帧和成片帧都过一遍 LUT（分析流不过，人脸/曝光按原始帧算）
            val previewUseCase = preview
            val analysisUseCase = imageAnalysis
            val captureUseCase = imageCapture
            val camera: Camera?
            if (previewUseCase != null && analysisUseCase != null && captureUseCase != null) {
                val groupBuilder = UseCaseGroup.Builder()
                    .addUseCase(previewUseCase)
                    .addUseCase(analysisUseCase)
                    .addUseCase(captureUseCase)
                lutEffect?.let { groupBuilder.addEffect(it) }
                camera = cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    selector,
                    groupBuilder.build()
                )
            } else {
                onError("相机组件未就绪")
                return
            }
            // ★ 拿到 CameraControl / CameraInfo，对焦、缩放、闪光灯、曝光补偿都要用
            cameraControl = camera?.cameraControl
            cameraInfo = camera?.cameraInfo
            // 切镜头后重建 ImageCapture，把用户选的闪光灯档位补回去
            imageCapture?.flashMode = flashMode
            // 重建会话会复位变焦与曝光补偿，把用户上次的选择补回去
            try {
                if (lastZoomRatio != 1f) cameraControl?.setZoomRatio(lastZoomRatio)
                if (lastEvIndex != 0) cameraControl?.setExposureCompensationIndex(lastEvIndex)
            } catch (e: Exception) {
                // 机型不支持或相机还没就绪，忽略
            }
        } catch (e: Exception) {
            onError("绑定相机失败: ${e.localizedMessage}")
        }
    }
}
