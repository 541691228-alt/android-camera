package cn.yege.dshcam

import org.junit.Assume
import org.junit.Test
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.io.RandomAccessFile
import java.nio.charset.StandardCharsets
import java.util.Locale

/**
 * 离线构图评测入口：把外部准备好的 64x48 像素按端上同一条链路跑一遍，导成 TSV 交给 Python 算指标。
 *
 * 为什么像素来自外部文件而不是这里自己解 JPEG：Android 模块的单元测试拿 android.jar 当编译期 JDK，
 * 里面没有 javax.imageio / java.awt（java.desktop 整个包都不在），Kotlin 侧根本读不了图。
 * 所以解码与缩放交给 tools/eval/pack_pixels.py，这边只负责"喂像素、跑算法、落表"。
 *
 * 写成单元测试而不是 main，是因为测试源集本来就看得见主源集那几个纯 Kotlin 算法类，
 * 不用再为评测另搭一套构建目标。
 */
class EvalHarnessTest {

    private data class EvalRow(val line: String, val status: String, val skipped: Boolean)

    /**
     * 评测用的目标宽高比。
     *
     * - 不设 / "src"：用原图宽高比，等价于"只变焦、不换比例"的裁剪；
     * - 给具体数（"1" 方构图、"0.75" 3:4 竖构图）：才真正考察换比例的重新构图。
     *
     * 两种口径的结论可能完全不一样，所以 EVAL_ASPECT 写错了不能悄悄退回 src —— 那样会得到
     * 一份看着正常、其实口径不对的报告。这里直接抛异常让本次评测失败。
     */
    private val evalAspect: Float? = run {
        val raw = System.getenv("EVAL_ASPECT")?.trim()
        when {
            raw.isNullOrEmpty() || raw.equals("src", ignoreCase = true) -> null
            else -> raw.toFloatOrNull()?.takeIf { it.isFinite() && it > 0f }
                ?: throw IllegalArgumentException("EVAL_ASPECT 只能是 src 或正数，实际是: $raw")
        }
    }

    /**
     * 显著度来源：给 EVAL_GRID 就用外部预生成的 64x48 网格，否则用规则算法现算。
     *
     * 线上优先吃 u2netp 的输出（SubjectModel.salienceGrid），规则算法（salienceFromArgb）
     * 只是模型不可用时的兜底 —— 两条路的显著图性格不同，同一个 bestCrop 的结果也会不同，
     * 所以评测必须能分开跑，报告里也要写明用的哪条。
     * 网格由 tools/eval/model_grid.py 按 SaliencyMath 的预处理/后处理逐行复刻产出。
     */
    private val evalGridPath: String? =
        System.getenv("EVAL_GRID")?.trim()?.takeIf { it.isNotEmpty() }

    @Test
    fun runEval() {
        val manifestPath: String? = System.getenv("EVAL_MANIFEST")
        val pixelsPath: String? = System.getenv("EVAL_PIXELS")
        val outPath: String? = System.getenv("EVAL_OUT")
        val gridPath = evalGridPath

        // 评测是显式触发的离线任务：环境变量没配就该跳过，
        // 否则常规 gradlew test 会莫名其妙多跑一遍全量图片
        Assume.assumeTrue(
            "设置 EVAL_MANIFEST / EVAL_OUT，以及 EVAL_PIXELS 或 EVAL_GRID 后才运行构图评测",
            !manifestPath.isNullOrEmpty() && !outPath.isNullOrEmpty() &&
                (!pixelsPath.isNullOrEmpty() || !gridPath.isNullOrEmpty())
        )

        val manifest = File(manifestPath!!)
        // 两个输入都是"每张 12288 字节的定长记录"，只是记录内容一个是 int32 像素、一个是 float32 网格。
        // 同时给时以 EVAL_GRID 为准：模型口径才是线上优先用的那条。
        val useGrid = !gridPath.isNullOrEmpty()
        val source = File(if (useGrid) gridPath!! else pixelsPath!!)
        // 缺输入说明这次不是评测运行（路径写错也算），跳过而不是报失败
        Assume.assumeTrue("清单文件不存在: ${manifest.absolutePath}", manifest.isFile)
        Assume.assumeTrue(
            if (useGrid) "显著度网格文件不存在: ${source.absolutePath}"
            else "像素文件不存在: ${source.absolutePath}",
            source.isFile
        )

        val outFile = File(outPath!!)
        outFile.parentFile?.mkdirs()

        var total = 0
        var ok = 0
        var skippedCount = 0
        var errors = 0

        RandomAccessFile(source, "r").use { raf ->
            BufferedWriter(
                OutputStreamWriter(FileOutputStream(outFile, false), StandardCharsets.UTF_8)
            ).use { writer ->
                writer.write(HEADER)
                writer.write("\n")
                writer.flush()

                for ((lineNo, raw) in manifest.readLines(StandardCharsets.UTF_8).withIndex()) {
                    val line = raw.trim().removePrefix("\uFEFF")
                    if (line.isEmpty() || line.startsWith("#")) continue
                    // 第一行是表头（index / image / w / h）
                    if (lineNo == 0 && line.substringBefore('\t').trim().lowercase() == "index") continue

                    total++
                    val row = runOne(raf, useGrid, line)
                    writer.write(row.line)
                    writer.write("\n")
                    // 每行都 flush：万一后面某张图把进程带崩，
                    // 前面已经算好的行必须还留在磁盘上
                    writer.flush()

                    when (row.status) {
                        "ok" -> {
                            ok++
                            if (row.skipped) skippedCount++
                        }
                        else -> errors++
                    }
                }
            }
        }

        println(
            "[eval-harness] total=$total ok=$ok skipped=$skippedCount errors=$errors " +
                "salience=${if (useGrid) "model-grid" else "rule-argb"} out=${outFile.path}"
        )
    }

    private fun runOne(raf: RandomAccessFile, useGrid: Boolean, line: String): EvalRow {
        val parts = line.split('\t')
        if (parts.size < 4) return errorRow(line, 0, 0, "error:bad_manifest_row")
        val index = parts[0].trim().toLongOrNull()
            ?: return errorRow(line, 0, 0, "error:bad_manifest_row")
        val path = parts[1].trim()
        val w = parts[2].trim().toIntOrNull() ?: 0
        val h = parts[3].trim().toIntOrNull() ?: 0
        if (w <= 0 || h <= 0) return errorRow(path, w, h, "error:bad_manifest_row")

        val offset = index * PIXEL_BYTES
        // 输入文件被截断时只标这一行，不要抛出去把整批判废
        if (offset < 0 || offset + PIXEL_BYTES > raf.length()) {
            return errorRow(path, w, h, "error:truncated")
        }

        return try {
            raf.seek(offset)
            if (useGrid) {
                // 模型显著度网格：64x48 个 float32（大端），由 model_grid.py 打包
                val grid = FloatArray(GRID_W * GRID_H)
                for (i in grid.indices) grid[i] = raf.readFloat()
                okRow(path, w, h, grid = grid)
            } else {
                val argb = IntArray(GRID_W * GRID_H)
                for (i in argb.indices) argb[i] = raf.readInt()
                okRow(path, w, h, argb = argb)
            }
        } catch (t: Throwable) {
            // 单张图上的算法异常要留痕但不能丢行，否则 Python 侧的计数会对不上清单
            errorRow(path, w, h, "error:" + t.javaClass.simpleName)
        }
    }

    private fun okRow(path: String, w: Int, h: Int, argb: IntArray? = null, grid: FloatArray? = null): EvalRow {
        // 端上喂给算法的是 64x48 位图、显著性网格也是 64x48（1 格 1 像素），
        // 这里保持同一口径，评测结果才有资格和线上比。
        // 有预生成网格就用网格（线上模型可用时的口径），否则用规则算法现算（兜底口径）。
        val s = if (grid != null) {
            AutoFrame.Salience(GRID_W, GRID_H, grid)
        } else {
            AutoFrame.salienceFromArgb(argb!!, GRID_W, GRID_H, GRID_W, GRID_H)
        }
        val subject = AutoFrame.subjectCenter(s)
        val guide = AutoFrame.guide(subject, levelOk = true, settled = false, trusted = false)

        // bestCrop 存在提前返回的分支，不会刷新 lastScoreInfo；
        // 先清空才能区分"这张没产生评分"和"残留了上一张的评分"
        AutoFrame.lastScoreInfo = ""
        // 目标宽高比：默认取原图比例（变焦口径），可用 EVAL_ASPECT 指定方/竖构图
        val aspect = evalAspect ?: (w.toFloat() / h.toFloat())
        // minKeep 取生产的 0.60f：默认值 0.70f 不是线上口径，用了评测会系统性偏保守
        val crop = AutoFrame.bestCrop(s, w, h, aspect, 0.60f)

        val keepArea = crop.width.toDouble() * crop.height / (w.toDouble() * h)
        // 端上 keep 面积够大就不另存 _auto 文件，这里复现同一个判定，方便对齐真实产出
        val skip = keepArea >= 0.97

        val f = ArrayList<String>(21)
        f.add(path)
        f.add(w.toString())
        f.add(h.toString())
        f.add(crop.left.toString())
        f.add(crop.top.toString())
        f.add(crop.width.toString())
        f.add(crop.height.toString())
        f.add(num(crop.score))
        f.add(num(keepArea))
        f.add(if (skip) "1" else "0")
        if (subject == null) {
            f.add("false")
            f.add("")
            f.add("")
            f.add("")
            f.add("")
        } else {
            f.add("true")
            f.add(num(subject.x))
            f.add(num(subject.y))
            f.add(num(subject.strength))
            f.add(num(subject.spread))
        }
        f.add(clean(guide.text))
        f.add(guide.ok.toString())
        f.add(guide.confident.toString())
        f.add(guide.framed.toString())
        f.add(clean(AutoFrame.lastScoreInfo))
        f.add("ok")

        return EvalRow(f.joinToString("\t"), "ok", skip)
    }

    private fun errorRow(path: String, w: Int, h: Int, status: String): EvalRow {
        val f = ArrayList<String>(21)
        f.add(path)
        f.add(w.toString())
        f.add(h.toString())
        f.add("0")
        f.add("0")
        f.add("0")
        f.add("0")
        f.add(num(0f))
        f.add(num(0.0))
        f.add("0")
        f.add("false")
        f.add("")
        f.add("")
        f.add("")
        f.add("")
        f.add("")
        f.add("false")
        f.add("false")
        f.add("false")
        f.add("")
        f.add(status)
        return EvalRow(f.joinToString("\t"), status, false)
    }

    // 显式指定 Locale：默认 Locale 在小数点/千分位上因环境而异，会让 TSV 在别的机器上解析失败
    private fun num(v: Float): String = String.format(Locale.US, "%.4f", v)

    private fun num(v: Double): String = String.format(Locale.US, "%.4f", v)

    // 制表符和换行会撕碎 TSV 的行列结构，落盘前一律压成空格
    private fun clean(s: String): String =
        s.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')

    private companion object {
        const val GRID_W = 64
        const val GRID_H = 48

        /** 每张图 64*48 个大端 int32 */
        const val PIXEL_BYTES = GRID_W * GRID_H * 4

        const val HEADER =
            "image\tw\th\tcrop_left\tcrop_top\tcrop_w\tcrop_h\tcrop_score\tkeep_area\tskipped\t" +
                "subject_found\tsubject_x\tsubject_y\tsubject_strength\tsubject_spread\t" +
                "guide_text\tguide_ok\tguide_confident\tguide_framed\tlast_score_info\tstatus"
    }
}
