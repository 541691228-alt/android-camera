package cn.yege.dshcam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapShader
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import android.content.res.Resources

/**
 * 取景覆盖层：三分线 / 水平仪 / 主体框 / 分数圆环 / 建议卡片 / 对焦框 /
 * 峰值对焦 / 斑马纹 / 构图引导（十字 + 箭头）。
 *
 * ★ 2026-10-05 界面重做（"专业相机风"）：
 *   · 统一四色：青 #00E5FF 辅助 / 绿 #4CAF50 好 / 橙 #FFC107 注意 / 红 #FF5252 差
 *   · 三分线改 0.8dp 的 #33FFFFFF（原来白色 1.5dp 太抢画面）
 *   · 分数从左上角小黑块改成右上角「圆环进度 + 大数字」
 *   · 建议带改成圆角卡片 + 左侧色条（按分数绿/黄/红）
 *   · 构图引导（AutoFrame.Guide）画：目标三分点十字 + 从主体指向目标的箭头，文字进卡片
 *
 * 这个 View 不消费触摸事件（没设 clickable），事件会落到下层的 PreviewView，
 * 所以点屏对焦 / 双指缩放手势照常工作。
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var facts: FrameFacts? = null
    private var advice: Advice? = null
    private var guide: AutoFrame.Guide? = null
    private var subject: AutoFrame.Subject? = null

    private val d = Resources.getSystem().displayMetrics.density
    private val sd = Resources.getSystem().displayMetrics.scaledDensity

    // 画笔初始化
    private val paintLine = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 0.8f * d
        color = Color.parseColor("#33FFFFFF")   // 三分线：很淡，别抢画面
    }

    private val paintHorizonFixed = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f * d
        color = Color.argb(102, 255, 255, 255)  // 白色 40%
    }

    private val paintHorizonInd = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * d
        color = COLOR_GOOD
    }

    private val paintFace = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * d
        color = COLOR_ACCENT
    }

    private val paintFaceLabel = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 12f * sd
        textAlign = Paint.Align.CENTER
    }

    // ★ 分数圆环（右上角）：底环 + 进度环 + 中心大数字
    private val paintRingBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f * d
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#40FFFFFF")
    }
    private val paintRing = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f * d
        strokeCap = Paint.Cap.ROUND
        color = COLOR_GOOD
    }
    private val paintScore = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 20f * sd
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
    }

    // ★ 建议卡片（底部）：圆角玻璃底 + 左侧色条 + 13sp 文字
    private val paintCardBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(166, 0, 0, 0)
    }
    private val paintCardBar = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = COLOR_GOOD
    }
    private val paintAdviceText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 13f * sd
        textAlign = Paint.Align.LEFT
    }
    private val paintGuideText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COLOR_ACCENT
        textSize = 13f * sd
        textAlign = Paint.Align.LEFT
        typeface = Typeface.DEFAULT_BOLD
    }

    private val paintHint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 14f * sd
        textAlign = Paint.Align.CENTER
    }

    // ★ 2026-10-04：白字画在过曝/白墙上完全看不见（实测 +2EV 时中央提示整条消失），
    //   所以给中央提示也垫一层半透明黑底（圆角，专业相机风的胶囊）
    private val paintHintBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(150, 0, 0, 0)
    }

    // ★ 2026-10-04 新增：点屏对焦的琥珀色对焦框
    //   原来用 #FFF176 + 1.5dp，实机在过曝/白墙画面里几乎看不见（+2EV 自检截图里整圈消失），
    //   改成更深的琥珀色 + 2.5dp
    private val paintFocus = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * d
        color = COLOR_WARN
    }

    // ★ 2026-10-05 新增：构图引导（目标点十字 + 箭头），青色
    private val paintGuide = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * d
        strokeCap = Paint.Cap.ROUND
        color = COLOR_ACCENT
    }

    // 复用矩形对象，避免 onDraw 中 new
    private val rectFace = RectF()
    private val rectHint = RectF()
    private val rectCard = RectF()
    private val rectBar = RectF()
    private val rectRing = RectF()
    private val rectSubject = RectF()
    private val rectTextBounds = Rect()

    // ★ 2026-10-04 新增：点屏对焦的焦点位置 + 1.2 秒后自动消失
    private var focusX = 0f
    private var focusY = 0f
    private var focusVisible = false
    private val hideFocusRunnable = Runnable {
        focusVisible = false
        invalidate()
    }

    /** 在 (x, y) 处闪一个对焦框，1.2 秒后自动消失（坐标＝本 View 的像素坐标） */
    fun showFocusRing(x: Float, y: Float) {
        focusX = x
        focusY = y
        focusVisible = true
        invalidate()
        removeCallbacks(hideFocusRunnable)
        postDelayed(hideFocusRunnable, 1200L)
    }

    /**
     * 每帧更新一次。
     * @param grid   峰值/斑马纹用的格子网格（合焦提示关着时为 null）
     * @param guide  构图引导（引导关着 / 没主体时为 null）
     * @param subject 本帧算出的主体质心（画引导箭头用，可为 null）
     */
    fun update(
        facts: FrameFacts,
        advice: Advice,
        grid: FocusAssist.AssistGrid? = null,
        guide: AutoFrame.Guide? = null,
        subject: AutoFrame.Subject? = null
    ) {
        this.facts = facts
        this.advice = advice
        this.guide = guide
        this.subject = subject
        // ★ 每帧都是新的网格对象 → 用对象身份判断要不要重算掩码（同一帧重绘时就不重算）
        if (grid !== assistGrid) {
            assistGrid = grid
            assistMaskId = -1
        }
        invalidate()
    }

    // ★ 2026-10-05 新增：峰值对焦（合焦处亮点）+ 斑马纹（过曝斜纹）
    //   assistMode: 0=关 1=只峰值 2=只斑马 3=两个都要
    private var assistMode = 0
    private var assistGrid: FocusAssist.AssistGrid? = null
    private var assistZebra: BooleanArray? = null
    private var assistPeak: BooleanArray? = null
    private var assistMaskId = -1
    private var peakPoints = FloatArray(0)
    private val rectAssist = RectF()
    private var zebraShader: BitmapShader? = null
    private val paintPeak = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = COLOR_ACCENT                       // 青色亮点：几乎所有场景都看得见
        style = Paint.Style.FILL
        strokeCap = Paint.Cap.ROUND
    }
    private val paintZebra = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    /** 切换辅助显示：0=关 1=峰值 2=斑马 3=都要 */
    fun setAssistMode(mode: Int) {
        assistMode = mode
        assistMaskId = -1
        invalidate()
    }

    /** ★ 构图引导开关：关掉就不画十字与箭头（数据侧也不再算显著性） */
    fun setGuideEnabled(on: Boolean) {
        guideEnabled = on
        if (!on) {
            guide = null
            subject = null
        }
        invalidate()
    }

    /** 斜纹贴图（斑马纹）：一张 2×2 格的 45° 斜纹小图，用 REPEAT 平铺 */
    private fun ensureZebraShader(d: Float) {
        if (zebraShader != null) return
        val cell = (13f * d).toInt().coerceAtLeast(6)
        val bmp = Bitmap.createBitmap(cell * 2, cell * 2, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        c.drawColor(Color.parseColor("#33FF5A00"))
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#D9FFFFFF")
            strokeWidth = cell * 0.5f
        }
        c.drawLine(-cell.toFloat(), cell * 2f, cell * 2f, -cell.toFloat(), p)
        c.drawLine(cell.toFloat(), cell * 3f, cell * 3f, cell.toFloat(), p)
        val sh = BitmapShader(bmp, Shader.TileMode.REPEAT, Shader.TileMode.REPEAT)
        zebraShader = sh
        paintZebra.shader = sh
    }

    /**
     * 峰值对焦 + 斑马纹的实际绘制（都在取景矩形内，按「格子」粒度）。
     * 掩码只在换网格或改模式时算一次；同一行连续的过曝格子合并成一个矩形画，省调用。
     */
    private fun drawAssist(canvas: Canvas, il: Float, itop: Float, iw: Float, ih: Float, d: Float) {
        val g = assistGrid ?: return
        if (g.cols <= 0 || g.rows <= 0 || g.luma.size < g.cols * g.rows) return
        val id = System.identityHashCode(g)
        if (assistMaskId != id) {
            assistZebra = if (assistMode and 2 != 0) FocusAssist.zebraMask(g) else null
            assistPeak = if (assistMode and 1 != 0) FocusAssist.peakingMask(g) else null
            assistMaskId = id
        }
        val cw = iw / g.cols
        val ch = ih / g.rows

        val zebra = assistZebra
        if (zebra != null && zebra.size >= g.cols * g.rows) {
            ensureZebraShader(d)
            for (r in 0 until g.rows) {
                var c = 0
                while (c < g.cols) {
                    if (zebra[r * g.cols + c]) {
                        val start = c
                        while (c < g.cols && zebra[r * g.cols + c]) c++
                        rectAssist.set(
                            il + start * cw, itop + r * ch,
                            il + c * cw, itop + (r + 1) * ch
                        )
                        canvas.drawRect(rectAssist, paintZebra)
                    } else {
                        c++
                    }
                }
            }
        }

        val peak = assistPeak
        if (peak != null && peak.size >= g.cols * g.rows) {
            paintPeak.strokeWidth = 3f * d
            val maxPts = 6000
            if (peakPoints.size < maxPts * 2) peakPoints = FloatArray(maxPts * 2)
            var n = 0
            outer@ for (r in 0 until g.rows) {
                for (c in 0 until g.cols) {
                    if (peak[r * g.cols + c]) {
                        if (n >= maxPts) break@outer
                        peakPoints[n * 2] = il + (c + 0.5f) * cw
                        peakPoints[n * 2 + 1] = itop + (r + 0.5f) * ch
                        n++
                    }
                }
            }
            if (n > 0) canvas.drawPoints(peakPoints, 0, n * 2, paintPeak)
        }
    }

    // ★ 2026-10-04 新增：与成片一致的「取景矩形」
    //   以前三分线 / 水平仪 / 人脸框都是按整个 View 画的，而 PreviewView 默认 FILL_CENTER
    //   会把 4:3 的画面裁掉两侧去铺满竖屏 —— 屏幕上的三分线跟照片里的三分线根本不在同一
    //   位置（对构图助手来说是最要命的错）。现在 PreviewView 改 FIT_CENTER，这里按成片
    //   宽高比算出画面矩形，所有取景元素都画在矩形内，HUD（分数/提示/建议）仍在全屏。
    //   imageAspect = 竖屏显示时的「宽 / 高」：4:3 → 0.75，16:9 → 0.5625，1:1 → 1.0
    private var imageAspect = 3f / 4f
    private var showGrid = true
    private var guideEnabled = true
    private val rectImage = RectF()

    fun setImageAspect(a: Float) {
        imageAspect = a
        invalidate()
    }

    fun setGridEnabled(on: Boolean) {
        showGrid = on
        invalidate()
    }

    private fun computeImageRect(w: Float, h: Float) {
        val viewAspect = w / h
        if (viewAspect > imageAspect) {          // View 比画面更宽 → 左右留黑边
            val iw = h * imageAspect
            rectImage.set((w - iw) / 2f, 0f, (w + iw) / 2f, h)
        } else {                                  // 竖屏常见：上下留黑边
            val ih = w / imageAspect
            rectImage.set(0f, (h - ih) / 2f, w, (h + ih) / 2f)
        }
    }

    /** 分数 → 颜色（绿/橙/红，与胶囊、卡片色条一致） */
    private fun scoreColor(score: Int): Int = when {
        score >= 85 -> COLOR_GOOD
        score >= 70 -> COLOR_WARN
        else -> COLOR_BAD
    }

    /**
     * 构图引导：目标三分点上画十字，从主体质心画一条带箭头的线指向它。
     * 达标（guide.ok）时只画一个绿色十字，不画箭头（画面干净）。
     */
    private fun drawGuide(canvas: Canvas, il: Float, itop: Float, iw: Float, ih: Float) {
        val g = guide ?: return
        // 没找到可信主体（画面里显著度铺满 / 太弱）时，不画十字和箭头 ——
        // 否则会拿弥散的质心当主体，永远催人往左往右（2026-10-04 用户反馈）。
        if (!g.confident) return
        val gx = il + g.targetX * iw
        val gy = itop + g.targetY * ih
        val arm = 9f * d
        val circleR = 13f * d
        paintGuide.color = if (g.framed) COLOR_GOOD else COLOR_ACCENT
        canvas.drawCircle(gx, gy, circleR, paintGuide)
        canvas.drawLine(gx - arm, gy, gx + arm, gy, paintGuide)
        canvas.drawLine(gx, gy - arm, gx, gy + arm, paintGuide)

        if (!g.framed) {
            val s = subject ?: return
            val sx = il + s.x * iw
            val sy = itop + s.y * ih
            // 箭头：从主体指向目标点，尾部留 14dp 不贴主体、头部留 16dp 不压十字
            var dx = gx - sx
            var dy = gy - sy
            val len = kotlin.math.sqrt(dx * dx + dy * dy)
            if (len < 40f * d) return
            dx /= len
            dy /= len
            val x0 = sx + dx * 14f * d
            val y0 = sy + dy * 14f * d
            val x1 = gx - dx * 18f * d
            val y1 = gy - dy * 18f * d
            canvas.drawLine(x0, y0, x1, y1, paintGuide)
            val head = 9f * d
            val ang = kotlin.math.atan2(dy.toDouble(), dx.toDouble())
            val a1 = ang + Math.PI * 0.82
            val a2 = ang - Math.PI * 0.82
            canvas.drawLine(x1, y1, (x1 + Math.cos(a1).toFloat() * head), (y1 + Math.sin(a1).toFloat() * head), paintGuide)
            canvas.drawLine(x1, y1, (x1 + Math.cos(a2).toFloat() * head), (y1 + Math.sin(a2).toFloat() * head), paintGuide)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()

        // 取景矩形（与成片一致）
        computeImageRect(w, h)
        val il = rectImage.left
        val itop = rectImage.top
        val ir = rectImage.right
        val ib = rectImage.bottom
        val iw = rectImage.width()
        val ih = rectImage.height()
        val icx = il + iw / 2f
        val icy = itop + ih / 2f

        // 1. 三分线九宫格（可开关，画在取景矩形内）
        if (showGrid) {
            val v1 = il + iw / 3f
            val v2 = il + iw * 2 / 3f
            val h1 = itop + ih / 3f
            val h2 = itop + ih * 2 / 3f
            canvas.drawLine(v1, itop, v1, ib, paintLine)
            canvas.drawLine(v2, itop, v2, ib, paintLine)
            canvas.drawLine(il, h1, ir, h1, paintLine)
            canvas.drawLine(il, h2, ir, h2, paintLine)
        }

        // 2. 水平仪（参考线与倾斜线都放在取景矩形里）
        canvas.drawLine(il, icy, ir, icy, paintHorizonFixed)
        val roll = facts?.rollDeg ?: 0f
        val levelOk = advice?.levelOk ?: false
        paintHorizonInd.color = if (levelOk) COLOR_GOOD else COLOR_WARN
        val indLen = iw * 0.4f
        canvas.save()
        canvas.translate(icx, icy)
        canvas.rotate(roll)
        canvas.drawLine(-indLen / 2f, 0f, indLen / 2f, 0f, paintHorizonInd)
        canvas.restore()

        // 3. 主体框（归一化坐标 × 取景矩形）
        val face = facts?.face
        if (face != null) {
            rectFace.set(
                il + face.left * iw,
                itop + face.top * ih,
                il + face.right * iw,
                itop + face.bottom * ih
            )
            canvas.drawRect(rectFace, paintFace)
            // 框上方文字
            val label = "主体"
            val labelH = paintFaceLabel.textSize
            val labelX = (rectFace.left + rectFace.right) / 2f
            val labelY = rectFace.top - labelH
            canvas.drawText(label, labelX, labelY, paintFaceLabel)
        } else {
            // 未识别提示（★ 2026-10-04 改：原来画在画面正中央，会跟点屏对焦的琥珀圈
            //   和中央水平线叠在一起（实机截图确认）；挪到顶部当状态条，构图区彻底干净。
            //   黑底是必须的：白墙/过曝场景下白字完全看不见；★ 2026-10-05 改圆角胶囊）
            val hint = "未识别到主体，把主体放进画面"
            val hintW = paintHint.measureText(hint)
            val hintH = paintHint.textSize
            val hintPadX = 16f * d
            val hintPadY = 8f * d
            val hintBaseline = h * 0.105f
            rectHint.set(
                w / 2f - hintW / 2f - hintPadX,
                hintBaseline - hintH - hintPadY,
                w / 2f + hintW / 2f + hintPadX,
                hintBaseline + hintPadY
            )
            canvas.drawRoundRect(rectHint, 16f * d, 16f * d, paintHintBg)
            canvas.drawText(hint, w / 2f, hintBaseline, paintHint)
        }

        // 3b. 峰值对焦 / 斑马纹（画在取景矩形内；关闭时整段跳过）
        if (assistMode != 0 && assistGrid != null) {
            drawAssist(canvas, il, itop, iw, ih, d)
        }

        // 3c. 构图引导：目标三分点十字 + 方向箭头
        if (guideEnabled) {
            drawGuide(canvas, il, itop, iw, ih)
        }

        // 4. ★ 分数圆环（右上角）：圆环进度 + 中间大数字
        val score = advice?.score ?: 0
        val col = scoreColor(score)
        val ringR = 24f * d
        val ringCx = w - 16f * d - ringR
        val ringCy = 16f * d + ringR
        rectRing.set(ringCx - ringR, ringCy - ringR, ringCx + ringR, ringCy + ringR)
        canvas.drawArc(rectRing, 0f, 360f, false, paintRingBg)
        paintRing.color = col
        canvas.drawArc(rectRing, -90f, 360f * (score.coerceIn(0, 100) / 100f), false, paintRing)
        paintScore.color = Color.WHITE
        // 数字基线：视觉上居中（字高约 0.72 × textSize）
        canvas.drawText(score.toString(), ringCx, ringCy + paintScore.textSize * 0.36f, paintScore)

        // 5. ★ 建议卡片（底部圆角卡片 + 左侧色条）
        //    · 中央提示已经说过「未识别到主体，把主体放进画面」，这里把同义那条去掉，别说两遍
        val lines = ArrayList<String>(3)
        val gText = guide?.text
        if (guideEnabled && !gText.isNullOrBlank()) lines.add(gText)
        (advice?.lines ?: emptyList())
            .filterNot { it.contains("未识别到主体") }
            .forEach { if (lines.size < 3) lines.add(it) }

        if (lines.isNotEmpty()) {
            val cardPad = 12f * d
            val barW = 5f * d
            val lineH = paintAdviceText.textSize + 4f * d
            val cardH = lineH * lines.size + cardPad * 2
            // 给底部控件（曝光面板 + 三个按钮 + 手势提示）留位置
            val bottomReserve = h * 0.20f
            val cardLeft = 14f * d
            val cardRight = w - 14f * d
            val cardBottom = h - bottomReserve
            val cardTop = cardBottom - cardH
            rectCard.set(cardLeft, cardTop, cardRight, cardBottom)
            canvas.drawRoundRect(rectCard, 14f * d, 14f * d, paintCardBg)

            // 左侧竖色条（按分数绿/黄/红；圆角跟着卡片）
            paintCardBar.color = col
            rectBar.set(cardLeft + 1f * d, cardTop + 1f * d, cardLeft + barW, cardBottom - 1f * d)
            canvas.drawRoundRect(rectBar, barW / 2f, barW / 2f, paintCardBar)

            var textY = cardTop + cardPad + paintAdviceText.textSize
            for (i in lines.indices) {
                val p = if (i == 0 && guideEnabled && !gText.isNullOrBlank()) paintGuideText else paintAdviceText
                canvas.drawText(lines[i], cardLeft + barW + 10f * d, textY, p)
                textY += lineH
            }
        }

        // 6. ★ 点屏对焦的琥珀色框（画在最上层，1.2 秒后自动消失）
        if (focusVisible) {
            val r = 36f * d
            val tick = 10f * d
            canvas.drawCircle(focusX, focusY, r, paintFocus)
            canvas.drawLine(focusX - r, focusY, focusX - r - tick, focusY, paintFocus)
            canvas.drawLine(focusX + r, focusY, focusX + r + tick, focusY, paintFocus)
            canvas.drawLine(focusX, focusY - r, focusX, focusY - r - tick, paintFocus)
            canvas.drawLine(focusX, focusY + r, focusX, focusY + r + tick, paintFocus)
        }
    }

    private companion object {
        /** 统一四色（与 res/values/colors.xml 的 cam_* 保持一致） */
        const val COLOR_ACCENT = 0xFF00E5FF.toInt()   // 青：辅助线 / 峰值 / 引导
        const val COLOR_GOOD = 0xFF4CAF50.toInt()     // 绿：好 / 达标
        const val COLOR_WARN = 0xFFFFC107.toInt()     // 橙：注意 / 对焦框
        const val COLOR_BAD = 0xFFFF5252.toInt()      // 红：差
    }
}
