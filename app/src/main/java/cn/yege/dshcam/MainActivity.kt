package cn.yege.dshcam

import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.core.app.ActivityCompat
import androidx.camera.view.PreviewView
import cn.yege.dshcam.CameraController
import cn.yege.dshcam.FaceAnalyzer
import cn.yege.dshcam.FrameObservation
import cn.yege.dshcam.FrameFacts
import cn.yege.dshcam.OverlayView
import cn.yege.dshcam.Rules
import cn.yege.dshcam.SensorPose
import cn.yege.dshcam.R
import java.util.Locale

class MainActivity : AppCompatActivity() {

    companion object {
        private const val REQUEST_CAMERA = 1001
        private const val FRAME_UPDATE_INTERVAL_MS = 120L
    }

    // 最新帧观测，volatile保证多线程可见性
    @Volatile
    private var latestObs: FrameObservation? = null

    // 视图组件
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var btnCapture: Button
    private lateinit var btnSwitch: Button
    private lateinit var btnFlash: Button
    private lateinit var seekEv: SeekBar
    private lateinit var tvEv: TextView
    private lateinit var tvZoom: TextView
    // ★ 2026-10-04 新增：左上角两个小胶囊（网格开关 / 画幅切换）
    private lateinit var chipGrid: TextView
    private lateinit var chipRatio: TextView
    private lateinit var chipStyle: TextView
    // ★ 2026-10-04 新增：自动曝光补偿开关（第 4 颗胶囊）
    private lateinit var chipAutoEv: TextView
    // ★ 2026-10-05 新增：合焦提示（峰值对焦 / 斑马纹）胶囊
    private lateinit var chipAssist: TextView
    // ★ 2026-10-05 新增：构图引导（A）+ 自动构图（B）两颗胶囊
    private lateinit var chipGuide: TextView
    private lateinit var chipAutoFrame: TextView

    // ★ 取景辅助状态
    private var gridOn = true
    private var ratioMode = CameraController.RATIO_43
    // ★ 实时调色风格序号（0=原图）
    private var styleIndex = LutStyles.ORIGINAL

    // ★ 自动曝光补偿：开关 + AutoExposure 的有符号防抖计数（正=连续想加曝光）
    private var autoEvOn = false
    private var autoEvStreak = 0
    /** 当前实际生效的曝光补偿（EV）。滑杆手动调与自动曝光共用这一个值。 */
    private var currentEv = 0f

    // ★ 合焦提示：0=关 1=峰值（合焦处亮点）2=斑马纹（过曝斜纹）3=两个都要
    private var assistMode = 0

    // ★ 构图引导（A）：实时告诉你往哪挪手机 / 端平；自动构图（B）：保存时另存一张自动裁好的
    private var guideOn = true
    private var autoFrameOn = true
    /** 最近一帧算出来的构图引导（null＝没主体或引导关着），交给覆盖层画十字+箭头 */
    private var currentGuide: AutoFrame.Guide? = null
    /** 上一帧是否已经"构图 OK"（给 AutoFrame.guide 做迟滞，避免容差边缘反复催） */
    private var guideSettled = false
    /** ★ 上一帧是否已经把主体当成可信主体（给 AutoFrame.guide 的强度门限做迟滞） */
    private var subjectTrusted = false
    /** 主体位置的平滑值（<0 表示还没有值）：帧间噪点会让方向和"还差多少%"乱跳 */
    private var smoothSx = -1f
    private var smoothSy = -1f
    /**
     * ★ 2026-10-05：强度/扩散也做平滑（<0＝还没值）。
     * 真机日志里规则网格的强度在 0.039 / 0.069 之间抖，正好骑在"有没有主体"的门限上，
     * 于是文案在「镜头往左移一点（约 13%）」和「没找到明显主体」之间一秒钟跳一次。
     * 位置平滑救不了它，得把这两个标量也一起平滑（0.6 旧 + 0.4 新）。
     */
    private var smoothStrengthVal = -1f
    private var smoothSpreadVal = -1f
    /** 上一帧的主体是不是来自人脸（用于在"人脸/显著性质心"两种来源切换时清掉平滑状态） */
    private var subjectFromFace = false
    /**
     * ★ 2026-10-05：上一次已并入平滑的「模型锚点」时刻。
     * 主体模型 1.5 秒才出一个锚点，同一个锚点被 30 帧反复并入平滑就等于没平滑
     * （数值会瞬间跳到锚点上），所以只在锚点**换新**的那个时刻做一次半步融合。
     */
    private var lastAnchorAt = 0L
    /** ★ 诊断用：引导来源日志的限流时刻（只在 `--ez guidedbg true` 时才会用到） */
    private var guideDbgAt = 0L
    /** ★ 诊断用：`--ez guidedbg true` 打开逐秒的引导诊断日志（默认关，别刷满 logcat） */
    private var guideDbgOn = false

    // ★ 2026-10-05：模型锚点的「锁定 + 迟滞」状态机（规则与真机日志见 SubjectLock.kt）。
    //   模型的位置不能直接用：杂乱场景里它每 1.5 秒换一个"主体"，提示就会横跳。
    private val modelLock = SubjectLock()

    // ★ 2026-10-04：曝光滑杆的范围要等相机 bind 完才知道，配好一次就不再试
    private var evReady = false
    // ★ 点屏对焦用：记按下位置，用来区分「单击」和「拖动/双指缩放」
    private var touchDownX = 0f
    private var touchDownY = 0f

    // 业务组件
    private lateinit var sensorPose: SensorPose
    private lateinit var faceAnalyzer: FaceAnalyzer
    private lateinit var cameraController: CameraController

    // 帧更新Handler
    private val mainHandler = Handler(Looper.getMainLooper())
    private val frameUpdateRunnable = object : Runnable {
        override fun run() {
            // 合成当前帧的事实数据
            val currentFacts = FrameFacts(
                viewW = overlayView.width,
                viewH = overlayView.height,
                face = latestObs?.faceNorm,
                faceCount = latestObs?.faceCount ?: 0,
                pitchDeg = sensorPose.pitchDeg(),
                rollDeg = sensorPose.rollDeg(),
                horizonY = null,
                meanLuma = latestObs?.meanLuma,
                overexposedRatio = latestObs?.overexposedRatio ?: 0f
            )

            // 计算构图建议并更新覆盖层（第三个参数＝峰值/斑马纹的格子网格，关着时为 null）
            val currentAdvice = Rules.evaluate(currentFacts)

            // ★ 构图引导（A）：用同一张格子网格算「显著性图」→ 主体质心 → 该往哪挪手机。
            //   网格只在 assistMode != 0 或 guideOn 时才算（FaceAnalyzer 的第二个闭包决定）。
            var guide: AutoFrame.Guide? = null
            var subject: AutoFrame.Subject? = null
            val assistGrid = latestObs?.assist
            val face = currentFacts.face
            // ★ 主体模型锚点（1.5 秒一个，见 FaceAnalyzer）：新鲜就优先于规则网格，过期就回退
            val anchor = if (::faceAnalyzer.isInitialized) faceAnalyzer.anchor else null
            val anchorFresh = anchor != null &&
                SystemClock.elapsedRealtime() - anchor.atMs < FaceAnalyzer.ANCHOR_TTL_MS
            // 主体来源在这两条之间切换时把平滑状态清零，免得脸和质心互相"拖"
            val useFace = face != null
            var guideSrc = "无"
            if (useFace != subjectFromFace) {
                smoothSx = -1f
                smoothSy = -1f
                smoothStrengthVal = -1f
                smoothSpreadVal = -1f
                subjectTrusted = false
                subjectFromFace = useFace
            }
            // ★ 模型锚点要先过"锁定"这一关（人脸在的时候不用它）：锁定失败就返回 null，
            //   让下面的规则网格分支接管 —— 规则路径的稳定性此前已在真机上验证过。
            val modelSubject = if (guideOn && !useFace && anchorFresh) resolveModelSubject(anchor) else null
            if (guideOn && useFace) {
                // ★ 2026-10-05：检测到人脸就直接拿脸当主体 —— 比"亮度/梯度质心"可靠得多
                //   （背景一亮点多，质心就被拽到画面中间，人明明站在边上）。
                //   spread 借用人脸高度占画面的比例：太小＝离得远（guide() 会提示靠近），
                //   太大＝脸快顶满（提示退半步）；strength 给 1 表示"这是可信主体"。
                val fx = (face!!.left + face.right) / 2f
                val fy = (face.top + face.bottom) / 2f
                val faceH = (face.bottom - face.top).coerceIn(0f, 1f)
                if (smoothSx < 0f) {
                    smoothSx = fx
                    smoothSy = fy
                } else {
                    smoothSx = smoothSx * 0.65f + fx * 0.35f
                    smoothSy = smoothSy * 0.65f + fy * 0.35f
                }
                subject = AutoFrame.Subject(smoothSx, smoothSy, 1f, faceH)
                guide = computeGuide(subject, currentAdvice.levelOk)
                lastAnchorAt = 0L
                guideSrc = "人脸"
            } else if (guideOn && modelSubject != null) {
                guideSrc = "模型锚点(已锁定)"
                // ★ 2026-10-05（用户选的"路线②"）：主体模型（u2netp）每 1.5 秒给一个锚点 ——
                //   它回答的是规则网格答不准的那个问题："这一帧里到底有没有主体"。
                //   但它的原始位置不能直接用：杂乱场景里它会在不同物体之间跳（真机日志见
                //   resolveModelSubject 的注释），所以只认"连续对得上"的锁定结果。
                subject = modelSubject
                guide = computeGuide(subject, currentAdvice.levelOk)
            } else if (guideOn && assistGrid != null) {
                val salience = AutoFrame.salienceFromGrid(
                    assistGrid.luma, assistGrid.grad, assistGrid.cols, assistGrid.rows
                )
                val raw = AutoFrame.subjectCenter(salience)
                if (raw != null) {
                    // 位置做一点平滑（0.65 旧 + 0.35 新）：帧间噪点会让方向和"还差多少%"来回跳
                    if (smoothSx < 0f) {
                        smoothSx = raw.x
                        smoothSy = raw.y
                    } else {
                        smoothSx = smoothSx * 0.65f + raw.x * 0.35f
                        smoothSy = smoothSy * 0.65f + raw.y * 0.35f
                    }
                    // 强度/扩散同样平滑：它们骑在"有没有主体"的门限上抖，文案就会一秒一跳
                    subject = AutoFrame.Subject(
                        smoothSx, smoothSy, smoothStrength(raw.strength), smoothSpread(raw.spread)
                    )
                }
                guide = computeGuide(subject, currentAdvice.levelOk)
                guideSrc = "规则网格"
            } else {
                smoothSx = -1f
                smoothSy = -1f
                smoothStrengthVal = -1f
                smoothSpreadVal = -1f
                guideSettled = false
                subjectTrusted = false
                subjectFromFace = false
                lastAnchorAt = 0L
                clearModelLock()
            }
            currentGuide = guide

            // 逐秒的引导诊断日志：默认关，`--ez guidedbg true` 时开（调引导/查横跳用）
            val nowDbg = SystemClock.elapsedRealtime()
            if (guideDbgOn && nowDbg - guideDbgAt >= 1000L) {
                guideDbgAt = nowDbg
                Log.d(
                    "CameraController",
                    "引导诊断：来源=$guideSrc 引导开=$guideOn 锚点新鲜=$anchorFresh 锚点=" +
                        (anchor?.let { "峰值%.3f/主体%c/age%ds".format(it.peak, if (it.hasSubject) '有' else '无', (nowDbg - it.atMs) / 1000) } ?: "无") +
                        " 主体=" + (subject?.let { "x=%.3f y=%.3f 强度=%.3f 扩散=%.3f".format(it.x, it.y, it.strength, it.spread) } ?: "无") +
                        " 文案=" + (guide?.text ?: "无")
                )
            }

            // 第 5 个参数＝主体质心，OverlayView 用它画「从主体指向目标三分点」的引导箭头
            overlayView.update(currentFacts, currentAdvice, assistGrid, guide, subject)

            // ★ 自动曝光补偿：画面持续偏暗/过曝时，攒够 5 帧同方向就自动走 1/3 EV 挡
            if (autoEvOn && evReady && ::cameraController.isInitialized) {
                val suggestion =
                    AutoExposure.suggest(currentFacts.meanLuma, currentFacts.overexposedRatio)
                val decision = AutoExposure.decide(suggestion, currentEv, autoEvStreak)
                autoEvStreak = decision.streak
                if (decision.changed) {
                    applyAutoEv(decision.ev)
                }
            }

            // ★ 变焦倍率指示（只在变化时改文本，不然每帧都触发布局）
            if (::cameraController.isInitialized) {
                val zoomText = String.format(Locale.US, "%.1f×", cameraController.currentZoom())
                if (tvZoom.text != zoomText) {
                    tvZoom.text = zoomText
                }
                // 曝光滑杆的范围要等相机 bind 完才知道，配好后 evReady 就不再进来
                if (!evReady) {
                    evReady = setupExposureSliderIfReady()
                }
            }

            // 循环执行更新
            mainHandler.postDelayed(this, FRAME_UPDATE_INTERVAL_MS)
        }
    }

    /**
     * ★ 2026-10-05：把「模型锚点」解析成一个可以交给引导的主体；返回 null＝这次不用模型，
     * 让规则网格分支接管。
     *
     * 为什么不能直接用模型给的位置：真机 19:30 的日志里，同一个杂乱桌面场景，u2netp 每 1.5 秒
     * 会在不同的东西上跳（玩具球 → 显示器 → RGB 风扇），锚点质心 x 从 0.276 跳到 0.541、
     * y 从 0.386 跳到 0.905，于是提示又变成「往右一点 / 压低一点 / 没找到主体」来回横跳
     * ——正是用户抱怨的那个毛病。锁定 + 迟滞的规则全都放在 [SubjectLock] 里（那样能单测）。
     */
    private fun resolveModelSubject(a: SubjectAnchor?): AutoFrame.Subject? {
        if (a == null) return null
        if (!modelLock.update(a)) return null
        // 同一个锚点会被后面若干帧反复读到，只在锚点**换新**的时刻做一次半步融合
        // （否则每帧都朝同一个值收敛＝看起来没平滑）
        if (lastAnchorAt != a.atMs) {
            if (smoothSx < 0f) {
                smoothSx = modelLock.x
                smoothSy = modelLock.y
            } else {
                smoothSx = smoothSx * 0.5f + modelLock.x * 0.5f
                smoothSy = smoothSy * 0.5f + modelLock.y * 0.5f
            }
            lastAnchorAt = a.atMs
        }
        return AutoFrame.Subject(smoothSx, smoothSy, a.strength, a.spread)
    }

    private fun clearModelLock() = modelLock.clear()

    /**
     * 统一算一次引导：把「已达标」「可信主体」两个迟滞状态一起喂进去并回写，
     * 免得三个分支各写一遍、漏掉一个就会让提示开始横跳。
     */
    private fun computeGuide(subject: AutoFrame.Subject?, levelOk: Boolean): AutoFrame.Guide {
        val g = AutoFrame.guide(subject, levelOk, guideSettled, subjectTrusted)
        guideSettled = g.framed
        subjectTrusted = g.confident
        return g
    }

    /** 强度/扩散的平滑（0.6 旧 + 0.4 新）——门限附近的抖动就是靠它压下去的，见字段注释 */
    private fun smoothStrength(next: Float): Float {
        smoothStrengthVal = if (smoothStrengthVal < 0f) next else smoothStrengthVal * 0.6f + next * 0.4f
        return smoothStrengthVal
    }

    private fun smoothSpread(next: Float): Float {
        smoothSpreadVal = if (smoothSpreadVal < 0f) next else smoothSpreadVal * 0.6f + next * 0.4f
        return smoothSpreadVal
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 绑定视图
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        btnCapture = findViewById(R.id.btnCapture)
        btnSwitch = findViewById(R.id.btnSwitch)
        btnFlash = findViewById(R.id.btnFlash)
        seekEv = findViewById(R.id.seekEv)
        tvEv = findViewById(R.id.tvEv)
        tvZoom = findViewById(R.id.tvZoom)
        chipGrid = findViewById(R.id.chipGrid)
        chipRatio = findViewById(R.id.chipRatio)
        chipStyle = findViewById(R.id.chipStyle)
        chipAutoEv = findViewById(R.id.chipAutoEv)
        chipAssist = findViewById(R.id.chipAssist)
        chipGuide = findViewById(R.id.chipGuide)
        chipAutoFrame = findViewById(R.id.chipAutoFrame)

        // ★ 网格开关：只影响覆盖层的三分线，随时可切
        chipGrid.setOnClickListener {
            gridOn = !gridOn
            overlayView.setGridEnabled(gridOn)
            chipGrid.text =
                getString(if (gridOn) R.string.chip_grid_on else R.string.chip_grid_off)
            refreshChipStyles()
        }

        // ★ 画幅切换：4:3 → 16:9 → 1:1 → 4:3（1:1 没有原生宽高比，拍完由相机控制器裁）
        chipRatio.setOnClickListener {
            val next = when (ratioMode) {
                CameraController.RATIO_43 -> CameraController.RATIO_169
                CameraController.RATIO_169 -> CameraController.RATIO_SQUARE
                else -> CameraController.RATIO_43
            }
            if (::cameraController.isInitialized) cameraController.setCaptureRatio(next)
            applyRatioUi(next)
        }
        applyRatioUi(CameraController.RATIO_43)

        // ★ 风格：原图 → 黑白 → 原图
        chipStyle.setOnClickListener {
            val next = (styleIndex + 1) % LutStyles.names.size
            if (::cameraController.isInitialized) cameraController.setLutStyle(next)
            applyStyleUi(next)
        }
        applyStyleUi(LutStyles.ORIGINAL)

        // ★ 自动曝光补偿开关：打开后每帧按 AutoExposure 的建议自动走 1/3 EV 挡（连续 5 帧同方向才动）
        chipAutoEv.setOnClickListener {
            autoEvOn = !autoEvOn
            autoEvStreak = 0
            chipAutoEv.text =
                getString(if (autoEvOn) R.string.chip_auto_ev_on else R.string.chip_auto_ev_off)
            refreshChipStyles()
        }

        // ★ 合焦提示：关 → 峰值 → 斑马 → 两个都要 → 关（画面里不花花绿绿的时候更多）
        chipAssist.setOnClickListener {
            assistMode = (assistMode + 1) % 4
            applyAssistUi(assistMode)
        }
        applyAssistUi(assistMode)

        // ★ 构图引导（A）：开着时覆盖层画三分点十字 + 指向箭头，并在卡片里给一句"往哪挪"
        chipGuide.setOnClickListener {
            guideOn = !guideOn
            if (!guideOn) currentGuide = null
            applyGuideUi()
        }
        applyGuideUi()

        // ★ 自动构图（B）：开着时保存会另存一张自动裁好的 _auto 版，原片不动
        chipAutoFrame.setOnClickListener {
            autoFrameOn = !autoFrameOn
            if (::cameraController.isInitialized) cameraController.setAutoFrameEnabled(autoFrameOn)
            applyAutoFrameUi()
        }
        applyAutoFrameUi()

        // 绑定按钮事件
        btnCapture.setOnClickListener {
            if (::cameraController.isInitialized) cameraController.capture()
        }
        btnSwitch.setOnClickListener {
            if (::cameraController.isInitialized) cameraController.switchLens()
        }
        // ★ 闪光灯：关 → 自动 → 开 → 关，按钮文字跟着变
        btnFlash.setOnClickListener {
            if (::cameraController.isInitialized) {
                btnFlash.text = flashLabel(cameraController.cycleFlash())
            }
        }
        // ★ 曝光补偿滑杆（范围/步长要等相机 bind 完，见 setupExposureSliderIfReady）
        seekEv.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!::cameraController.isInitialized) return
                val range = cameraController.exposureRange() ?: return
                val index = range.lower + progress
                cameraController.setExposureIndex(index)
                currentEv = index * cameraController.exposureStep()
                autoEvStreak = 0     // 用户自己动过曝光，自动曝光的累计作废、重新攒
                tvEv.text = String.format(
                    Locale.US, "曝光 %+.1f EV", currentEv
                )
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {}

            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })
        // ★ 手势：点屏对焦 + 双指缩放
        setupGestures()

        // 申请相机权限
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                REQUEST_CAMERA
            )
        } else {
            initBusinessComponents()
        }
    }

    // ------------------------------------------------------------------
    // ★ 2026-10-04 新增：手势（点屏对焦 / 双指缩放）、闪光灯文字、曝光滑杆
    // ------------------------------------------------------------------

    /** 闪光灯档位 → 按钮文字 */
    private fun flashLabel(mode: Int): String = when (mode) {
        ImageCapture.FLASH_MODE_AUTO -> getString(R.string.btn_flash_auto)
        ImageCapture.FLASH_MODE_ON -> getString(R.string.btn_flash_on)
        else -> getString(R.string.btn_flash_off)
    }

    /**
     * 画幅 → 胶囊文字 + 覆盖层的取景矩形宽高比。
     * 覆盖层所有取景元素（三分线/水平仪/人脸框）都画在这个矩形里，保证「看到的就是拍到的」。
     * 竖屏显示时的宽高比：4:3 → 0.75，16:9 → 0.5625，1:1 → 1.0
     */
    private fun applyRatioUi(mode: Int) {
        ratioMode = mode
        val label: Int
        val aspect: Float
        when (mode) {
            CameraController.RATIO_169 -> {
                label = R.string.chip_ratio_169
                aspect = 9f / 16f
            }
            CameraController.RATIO_SQUARE -> {
                label = R.string.chip_ratio_11
                aspect = 1f
            }
            else -> {
                label = R.string.chip_ratio_43
                aspect = 3f / 4f
            }
        }
        chipRatio.text = getString(label)
        overlayView.setImageAspect(aspect)
        refreshChipStyles()
    }

    /** 风格序号 → 胶囊文字（名字来自 LutStyles.names，0 是原图） */
    private fun applyStyleUi(index: Int) {
        styleIndex = index
        val name = LutStyles.names.getOrElse(index) { LutStyles.names.first() }
        chipStyle.text = getString(R.string.chip_style_prefix, name)
        refreshChipStyles()
    }

    /**
     * 点屏对焦 + 双指缩放。
     * 手势挂在 PreviewView 上：构图覆盖层 OverlayView 不消费触摸事件（它没设 clickable），
     * 事件会从上层落回 PreviewView；底部按钮在自己的范围内优先吃掉点击。
     */
    private fun setupGestures() {
        val scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    if (::cameraController.isInitialized) {
                        cameraController.zoomBy(detector.scaleFactor)
                    }
                    return true
                }
            }
        )
        scaleDetector.isQuickScaleEnabled = false

        previewView.setOnTouchListener { _, event ->
            if (!::cameraController.isInitialized) {
                return@setOnTouchListener false
            }
            scaleDetector.onTouchEvent(event)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    touchDownX = event.x
                    touchDownY = event.y
                }
                MotionEvent.ACTION_UP -> {
                    val moved = kotlin.math.hypot(
                        (event.x - touchDownX).toDouble(),
                        (event.y - touchDownY).toDouble()
                    )
                    // 只有「没在缩放、移动距离也小」的单击才当对焦，避免滑动误触发
                    if (!scaleDetector.isInProgress && moved < 40.0) {
                        val w = previewView.width
                        val h = previewView.height
                        if (w > 0 && h > 0) {
                            cameraController.focusAt(event.x / w, event.y / h)
                            overlayView.showFocusRing(event.x, event.y)
                        }
                    }
                }
            }
            true
        }
    }

    /**
     * 配置曝光补偿滑杆：范围来自机型的 ExposureState（通常 ±12 档、1/3 EV 一档）。
     * 相机还没 bind 好时返回 false，帧循环会一直重试到成功。
     */
    private fun setupExposureSliderIfReady(): Boolean {
        if (!::cameraController.isInitialized) return false
        val range = cameraController.exposureRange() ?: return false
        seekEv.max = (range.upper - range.lower).coerceAtLeast(1)
        seekEv.progress = (-range.lower).coerceIn(0, seekEv.max)   // 中间 = 0 EV
        cameraController.setExposureIndex(0)
        currentEv = 0f
        tvEv.text = String.format(Locale.US, "曝光 %+.1f EV", 0f)
        return true
    }

    /**
     * 把自动曝光算出的 EV 落到相机和滑杆上。
     *
     * 坑（第一版写错、被真机打脸）：相机接口收的是「曝光补偿档位序号」index，
     * 而 index 是**绝对档位**——index 0 就是 0 EV（Camera2 的约定），不是「相对 range.lower 的偏移」。
     * 所以 index 必须直接由 EV 除以每档 EV 得到：ev 0.33 / step 0.333 = index 1。
     * 第一版写成 `range.lower + Math.round(ev / step)`，等于把每个目标档位又整体平移了 range.lower
     * （本机是 -12）→ 想要 +0.33 EV 反而落到 index -11（≈ -3.7 EV），最后被夹到最低档 -12（-4.0 EV）：
     * 画面越暗、它越往暗里推，正是真机上看到的「黑屏 + 曝光 -4.0 EV」。
     * 滑杆那条线用的是相对位置（progress = index - range.lower），两者别混。
     */
    private fun applyAutoEv(ev: Float) {
        val range = cameraController.exposureRange() ?: return
        val step = cameraController.exposureStep()
        if (step <= 0f) return
        val index = Math.round(ev / step).coerceIn(range.lower, range.upper)
        cameraController.setExposureIndex(index)
        currentEv = index * step
        seekEv.progress = (index - range.lower).coerceIn(0, seekEv.max)
        tvEv.text = String.format(Locale.US, "曝光 %+.1f EV", currentEv)
    }

    /**
     * 合焦提示胶囊的显示与状态。
     * 0=关（默认，画面干净）1=峰值（合焦处点亮青色小点）2=斑马（过曝处 45° 斜纹）3=两个都要。
     * 关掉时 [FaceAnalyzer] 连格子网格都不算，等于零开销。
     */
    private fun applyAssistUi(mode: Int) {
        assistMode = mode
        overlayView.setAssistMode(mode)
        chipAssist.text = getString(
            when (mode) {
                1 -> R.string.chip_assist_peak
                2 -> R.string.chip_assist_zebra
                3 -> R.string.chip_assist_both
                else -> R.string.chip_assist_off
            }
        )
        refreshChipStyles()
    }

    /** 胶囊底色：开着/生效 = 青底描边，关着 = 半透明黑底 */
    private fun chipBackground(on: Boolean): Int =
        if (on) R.drawable.bg_chip_on else R.drawable.bg_chip

    /**
     * 所有胶囊的选中态统一在这里刷。
     * 专业相机风的要求：一眼能看出「哪些辅助开着」，所以不靠文字变化、靠底色。
     */
    private fun refreshChipStyles() {
        chipGrid.setBackgroundResource(chipBackground(gridOn))
        chipRatio.setBackgroundResource(chipBackground(ratioMode != CameraController.RATIO_43))
        chipStyle.setBackgroundResource(chipBackground(styleIndex != LutStyles.ORIGINAL))
        chipAutoEv.setBackgroundResource(chipBackground(autoEvOn))
        chipGuide.setBackgroundResource(chipBackground(guideOn))
        chipAutoFrame.setBackgroundResource(chipBackground(autoFrameOn))
        chipAssist.setBackgroundResource(chipBackground(assistMode != 0))
    }

    /** 构图引导（A）开关：文字 + 底色 + 通知覆盖层画/不画十字与箭头 */
    private fun applyGuideUi() {
        chipGuide.text =
            getString(if (guideOn) R.string.chip_guide_on else R.string.chip_guide_off)
        overlayView.setGuideEnabled(guideOn)
        refreshChipStyles()
    }

    /** 自动构图（B）开关：文字 + 底色（保存时是否另存 _auto 版由 CameraController 决定） */
    private fun applyAutoFrameUi() {
        chipAutoFrame.text =
            getString(if (autoFrameOn) R.string.chip_autoframe_on else R.string.chip_autoframe_off)
        refreshChipStyles()
    }

    // 初始化业务组件
    private fun initBusinessComponents() {
        // 初始化传感器姿态监听
        sensorPose = SensorPose(this)
        sensorPose.start()

        // 初始化人脸分析器（第二个参数 = 是否要算峰值对焦/斑马纹的格子网格）
        // ★ 构图引导（A）也要用这张网格算「显著性」，所以引导开着时同样要算
        faceAnalyzer = FaceAnalyzer(
            { observation -> latestObs = observation },
            { assistMode != 0 || guideOn }
        )
        // 逐锚点/逐秒的细节日志默认关，`--ez guidedbg true` 时开（调引导用的）
        faceAnalyzer.debug = guideDbgOn

        // 初始化相机控制器
        cameraController = CameraController(
            context = this,
            lifecycleOwner = this,
            previewView = previewView,
            analyzerFactory = { faceAnalyzer },
            onError = { errorMsg ->
                runOnUiThread {
                    Toast.makeText(this, errorMsg, Toast.LENGTH_SHORT).show()
                }
            },
            onSaved = {
                runOnUiThread {
                    Toast.makeText(this, R.string.saved, Toast.LENGTH_SHORT).show()
                }
            }
        )
        // ★ 2026-10-05：主体模型（u2netp 4.4 MB）在后台先加载好，别让用户第一次按快门
        //    时才付那几百毫秒；加载失败会在拍照时静默回退到规则算法。
        Thread {
            try {
                SubjectModel.ensureLoaded(applicationContext)
            } catch (_: Throwable) {
            }
        }.start()

        // 调试钩子3（远程验证画幅 / 网格，小米禁 adb 注入点击只能这么验）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ei ratio 1    (0=4:3 1=16:9 2=1:1)
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ez grid false (关掉三分线)
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ez guidedbg true
        //     （打开逐锚点/逐秒的引导诊断日志，默认关；查"提示横跳"和"锁没锁上主体"用）
        guideDbgOn = intent?.getBooleanExtra("guidedbg", false) == true
        if (::faceAnalyzer.isInitialized) faceAnalyzer.debug = guideDbgOn
        val dbgRatio = intent?.getIntExtra("ratio", -1) ?: -1
        if (dbgRatio in CameraController.RATIO_43..CameraController.RATIO_SQUARE) {
            cameraController.setCaptureRatio(dbgRatio)
            applyRatioUi(dbgRatio)
        }
        if (intent?.hasExtra("grid") == true) {
            gridOn = intent.getBooleanExtra("grid", true)
            overlayView.setGridEnabled(gridOn)
            chipGrid.text =
                getString(if (gridOn) R.string.chip_grid_on else R.string.chip_grid_off)
            refreshChipStyles()
        }
        // 调试钩子4（远程验证实时调色）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ei style 1  (0=原图 / 1=黑白)
        val dbgStyle = intent?.getIntExtra("style", -1) ?: -1
        if (dbgStyle in 0 until LutStyles.names.size) {
            cameraController.setLutStyle(dbgStyle)
            applyStyleUi(dbgStyle)
        }

        // 调试钩子6（主体模型自检：不需要相机、不需要亮屏，直接拿手机上的图跑一次 u2netp）：
        //   adb push 图 /sdcard/Android/data/cn.yege.dshcam/files/t.jpg
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --es modeltest /sdcard/Android/data/cn.yege.dshcam/files/t.jpg
        // 打三遍取平均值（顺手量手机上真实推理耗时），日志 tag 是 CameraController。
        val modelTestPath = intent?.getStringExtra("modeltest")
        if (!modelTestPath.isNullOrEmpty()) {
            Thread {
                try {
                    val bmp = android.graphics.BitmapFactory.decodeFile(modelTestPath)
                    val cols = 64
                    val rows = 48
                    if (bmp == null) {
                        Log.w("CameraController", "主体模型自检：解不开 $modelTestPath")
                    } else if (!SubjectModel.ensureLoaded(applicationContext)) {
                        Log.w("CameraController", "主体模型自检：模型加载失败")
                    } else {
                        var total = 0L
                        var grid: FloatArray? = null
                        repeat(3) { i ->
                            val t0 = System.currentTimeMillis()
                            val g = SubjectModel.salienceGrid(bmp, cols, rows)
                            total += System.currentTimeMillis() - t0
                            if (i == 0) grid = g
                        }
                        val g = grid
                        if (g == null) {
                            Log.w("CameraController", "主体模型自检：推理失败")
                        } else {
                            val c = AutoFrame.subjectCenter(AutoFrame.Salience(cols, rows, g))
                            val where = if (c == null) "无" else
                                "x=%.3f y=%.3f strength=%.3f spread=%.3f".format(
                                    c.x, c.y, c.strength, c.spread
                                )
                            Log.d(
                                "CameraController",
                                "主体模型自检 ${bmp.width}×${bmp.height}：峰值=${"%.3f".format(SubjectModel.lastPeak)}" +
                                    " 平均耗时=${total / 3}ms 主体=$where 判据=" +
                                    (if (SubjectModel.lastPeak >= SaliencyMath.GRID_PEAK_GATE) "有主体" else "没有主体")
                            )
                        }
                        bmp.recycle()
                    }
                } catch (t: Throwable) {
                    Log.w("CameraController", "主体模型自检失败：${t.localizedMessage}")
                }
            }.start()
        }

        // 调试钩子5（复现「点风格胶囊后预览变黑」，远程验证用；小米禁 adb 注入点击）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ei style 0 --ei style2 1
        // 启动 5 秒后走和用户点胶囊完全相同的路径切一次风格。
        val dbgStyle2 = intent?.getIntExtra("style2", -1) ?: -1
        if (dbgStyle2 in 0 until LutStyles.names.size) {
            mainHandler.postDelayed({
                if (::cameraController.isInitialized) {
                    cameraController.setLutStyle(dbgStyle2)
                    applyStyleUi(dbgStyle2)
                }
            }, 5000L)
        }

        cameraController.start()

        // ★ 自动构图（B）默认开着：让控制器知道保存时要不要另存一张 _auto 版
        cameraController.setAutoFrameEnabled(autoFrameOn)

        // 调试钩子6（远程验证自动曝光，小米禁 adb 注入点击）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ez autoev true
        // 启动即把「自动EV」打开，和用户点胶囊走同一个状态。
        if (intent?.getBooleanExtra("autoev", false) == true) {
            autoEvOn = true
            autoEvStreak = 0
            chipAutoEv.text = getString(R.string.chip_auto_ev_on)
        }

        // 调试钩子7（远程验证「EV → 相机档位」换算，小米禁 adb 注入点击）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ei evtest 3
        // 参数是 EV×10（3 = +0.3 EV、-40 = -4.0 EV），启动 3 秒后直接调 applyAutoEv()。
        // 默认 0 表示不测。配合 --ez autoev true 用：先用 evtest 把曝光压到 -4.0 EV 造一个「黑画面」，
        // 看自动曝光会不会自己把它拉回来（修复前会一路顶到最低档不动）。
        val evTest = intent?.getIntExtra("evtest", 0) ?: 0
        if (evTest != 0) {
            mainHandler.postDelayed({ applyAutoEv(evTest / 10f) }, 3000L)
        }

        // 调试钩子8（远程验证合焦提示，小米禁 adb 注入点击）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ei assist 3
        // 0=关 1=只峰值 2=只斑马 3=两个都要
        val dbgAssist = intent?.getIntExtra("assist", -1) ?: -1
        if (dbgAssist in 0..3) {
            applyAssistUi(dbgAssist)
        }

        // 调试钩子9（远程验证构图引导 A / 自动构图 B，小米禁 adb 注入点击）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ei guide 0 --ei autoframe 1
        // guide/autoframe 都是 0/1；不传就不动默认值（两个默认都是开）。
        val dbgGuide = intent?.getIntExtra("guide", -1) ?: -1
        if (dbgGuide in 0..1) {
            guideOn = dbgGuide == 1
            if (!guideOn) currentGuide = null
            applyGuideUi()
        }
        val dbgAutoFrame = intent?.getIntExtra("autoframe", -1) ?: -1
        if (dbgAutoFrame in 0..1) {
            autoFrameOn = dbgAutoFrame == 1
            cameraController.setAutoFrameEnabled(autoFrameOn)
            applyAutoFrameUi()
        }
        // 调试钩子10：--ez forcetrim true
        //   "自动构图"判定当前画面不用裁时也强制裁一刀，用来验证另存链路
        //   （MediaStore 写入 + EXIF 方向 + 相册扫描）真的能出 _auto.jpg。
        if (intent?.getBooleanExtra("forcetrim", false) == true) {
            cameraController.setDebugForceTrim(true)
        }

        // 启动帧更新循环
        mainHandler.post(frameUpdateRunnable)

        // 调试钩子（远程验证用）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ez capture true
        // 启动 3 秒后自动按一次快门。小米默认禁止 adb 注入点击
        // （input tap 会报 SecurityException: INJECT_EVENTS），只能靠这个钩子验「拍照→存图」。
        if (intent?.getBooleanExtra("capture", false) == true) {
            mainHandler.postDelayed({
                if (::cameraController.isInitialized) {
                    cameraController.capture()
                }
            }, 3000L)
        }

        // 调试钩子2（四件套自检，同为远程验证用）：
        //   adb shell am start -n cn.yege.dshcam/.MainActivity --ez selftest true
        // 启动 3 秒后自动：曝光 +2.0EV、闪光灯切到「自动」、对焦屏幕中心、变焦 1.6×。
        // 小米默认禁止 adb 注入点击（input tap 报 INJECT_EVENTS），只能靠这个钩子验
        // 「点屏对焦 / 双指缩放 / 闪光灯 / 曝光补偿」四条链路。
        if (intent?.getBooleanExtra("selftest", false) == true) {
            mainHandler.postDelayed({
                if (::cameraController.isInitialized) {
                    val range = cameraController.exposureRange()
                    val step = cameraController.exposureStep()
                    if (range != null) {
                        val index = (2f / step).toInt().coerceIn(range.lower, range.upper)
                        cameraController.setExposureIndex(index)
                        seekEv.progress = index - range.lower
                        tvEv.text = String.format(Locale.US, "曝光 %+.1f EV", index * step)
                    }
                    btnFlash.text = flashLabel(cameraController.cycleFlash())  // 关 → 自动
                    cameraController.zoomBy(1.6f)
                    cameraController.focusAt(0.5f, 0.5f)
                    overlayView.showFocusRing(
                        overlayView.width / 2f,
                        overlayView.height / 2f
                    )
                }
            }, 3000L)
        }
    }

    // 权限回调处理
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initBusinessComponents()
            } else {
                Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (::cameraController.isInitialized) {
            cameraController.start()
            sensorPose.start()
            mainHandler.post(frameUpdateRunnable)
        }
    }

    override fun onPause() {
        super.onPause()
        mainHandler.removeCallbacksAndMessages(null)
        if (::cameraController.isInitialized) {
            cameraController.stop()
            sensorPose.stop()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mainHandler.removeCallbacksAndMessages(null)
        if (::sensorPose.isInitialized) {
            sensorPose.stop()
        }
        if (::cameraController.isInitialized) {
            cameraController.stop()
        }
        if (::faceAnalyzer.isInitialized) {
            faceAnalyzer.release()
        }
    }
}
