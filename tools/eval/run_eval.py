#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构图算法离线评测脚本。
读取 manifest 和 harness 数据，调用 metrics 模块计算指标，生成 JSON 和 Markdown 报告。
"""

import argparse
import csv
import json
import os
import sys
import statistics

import numpy as np

# 确保能导入同目录下的 metrics 模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import metrics


def parse_args():
    """解析命令行参数。默认路径都相对当前目录的 work/，报告目录跟着 --manifest 走。"""
    parser = argparse.ArgumentParser(description="构图算法离线评测")
    parser.add_argument("--manifest", type=str, default=os.path.join("work", "manifest.tsv"),
                        help="Manifest TSV 路径")
    parser.add_argument("--harness", type=str, default=os.path.join("work", "harness_out.tsv"),
                        help="Harness 输出 TSV 路径")

    # 报告默认落到 --manifest 所在目录。这里必须用刚解析到的实际值：
    # 拿默认值算的话，用户换了一个 --manifest，报告还会写回默认目录。
    args, _ = parser.parse_known_args()
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    parser.add_argument("--out-json", type=str, default=os.path.join(manifest_dir, "report.json"),
                        help="JSON 报告路径")
    parser.add_argument("--out-md", type=str, default=os.path.join(manifest_dir, "report.md"),
                        help="Markdown 报告路径")
    parser.add_argument("--label", type=str, default="",
                        help="本次评测口径的名字（比如 src / 1:1 / 3:4），会写进报告标题和 JSON")
    parser.add_argument("--aspect", type=str, default="src",
                        help="仅用于记录：harness 这次用的 EVAL_ASPECT（src 或具体数值）")

    return parser.parse_args()


def parse_bool_int(value, default=0):
    """harness 的布尔列写的是 true/false（不是 1/0）。

    解析不出来就抛 ValueError，让上层按"该行格式有问题"计入 errors —— 静默当成 False
    会让 subject_found_rate 悄悄偏低，这种错最难查。
    """
    if value is None:
        return default
    s = str(value).strip().lower()
    if s == "":
        return default
    if s in ("1", "true"):
        return 1
    if s in ("0", "false"):
        return 0
    raise ValueError("不是布尔值: %r" % (value,))


def load_manifest(path):
    """加载 manifest 文件，返回 {image_path: mask_path} 字典。"""
    data = {}
    # utf-8-sig：Windows 上存的清单常带 BOM，带进第一列列名会让整份文件读不出来
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError("Manifest 文件格式错误")
        
        # 假设第一列是 image，第二列是 mask
        for row in reader:
            if len(row) >= 2:
                img_path = row[0].strip()
                mask_path = row[1].strip()
                data[img_path] = mask_path
    return data


def load_harness(path):
    """加载 harness 文件，返回 {image_path: row_dict} 字典及错误计数。"""
    data = {}
    errors = 0
    
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    # 校验必要字段是否存在且可解析
                    if "image" not in row or "status" not in row:
                        errors += 1
                        continue
                    
                    img_path = row["image"].strip()
                    status = row.get("status", "").strip().lower()
                    
                    # 只有 status == "ok" 且 crop 面积 > 0 才视为有效行
                    # 这里先存下来，后续再过滤，或者现在过滤。
                    # 为了统计 errors，我们在这里判断是否“格式正确但逻辑无效”
                    # 规格说：只有 status=="ok" 且 crop_w*crop_h>0 的行参与评测，其它行计入 errors。
                    # 所以如果 status != ok 或 crop 无效，直接算 errors，不存入 data。
                    
                    # 尝试解析数值
                    w = int(row.get("w", 0))
                    h = int(row.get("h", 0))
                    cl = int(row.get("crop_left", 0))
                    ct = int(row.get("crop_top", 0))
                    cw = int(row.get("crop_w", 0))
                    ch = int(row.get("crop_h", 0))
                    keep_area = float(row.get("keep_area", 0.0))
                    skipped = int(row.get("skipped", 0))
                    subject_found = parse_bool_int(row.get("subject_found", ""))
                    
                    if status != "ok" or (cw * ch <= 0):
                        errors += 1
                        continue
                        
                    data[img_path] = {
                        "w": w, "h": h,
                        "crop": (cl, ct, cw, ch),
                        "keep_area": keep_area,
                        "skipped": skipped,
                        "subject_found": subject_found
                    }
                except (ValueError, TypeError):
                    errors += 1
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Harness 文件不存在：{path}")
        
    return data, errors


def calc_stats(values):
    """计算列表的均值和中位数，空列表返回 0.0。"""
    if not values:
        return 0.0, 0.0
    mean_val = float(np.mean(values))
    try:
        median_val = float(statistics.median(values))
    except statistics.StatisticsError:
        median_val = 0.0
    return mean_val, median_val


def main():
    args = parse_args()
    
    # 加载数据
    try:
        manifest = load_manifest(args.manifest)
    except Exception as e:
        print(f"错误：无法加载 Manifest: {e}")
        sys.exit(1)
        
    try:
        harness, n_errors = load_harness(args.harness)
    except Exception as e:
        print(f"错误：无法加载 Harness: {e}")
        sys.exit(1)
        
    # 按图像路径排序，保证确定性
    sorted_images = sorted(manifest.keys())
    
    # 统计变量
    n_images = 0
    no_mask_count = 0
    
    # 收集各模式的指标
    pred_ks, pred_ds = [], []
    center_ks, center_ds = [], []
    optimal_ks, optimal_ds = [], []
    whole_ks, whole_ds = [], []
    
    # Win/Tie/Lose
    wins, ties, loses = 0, 0, 0
    # 胜负幅度累加器（赢的时候 / 输的时候各平均差多少保留率）
    win_gain, lose_drop = 0.0, 0.0
    
    # Capture Ratio 累加器
    cap_num, cap_den = 0.0, 0.0
    
    # Skip & Subject
    skipped_count = 0
    subject_found_count = 0
    keep_areas = []

    # 构图质量：主体在裁剪框里的位置
    # 保留率只管"有没有把主体裁掉"，管不了"主体摆得正不正"，所以另外收一组：
    #   离三分点距离（越小越接近三分构图）、摆正中的比例（贴在框中心）、切到主体的比例。
    pred_thirds, center_thirds = [], []
    pred_center_d, center_center_d = [], []
    pred_dead, center_dead = 0, 0
    pred_cut, center_cut = 0, 0
    DEAD_CENTER_TOL = 0.05  # 主体质心离框中心 5% 以内就算"摆正中"
    CUT_TOL = 0.99          # 框内不到 99% 的主体像素就算切到了主体
    # 主体损失分档：lost = 1 - 保留率，也就是被裁掉的主体像素比例。
    # 只看"有没有切到"（CUT_TOL）分不出"掉 1%"和"掉一半"，这里给一条严重度曲线。
    LOSS_LEVELS = (0.001, 0.01, 0.05, 0.10, 0.20, 0.50)
    
    # By Size Bins
    bins = {
        ">=0.97": {"count": 0, "pred_ks": [], "center_ks": []},
        "0.8~0.97": {"count": 0, "pred_ks": [], "center_ks": []},
        "0.5~0.8": {"count": 0, "pred_ks": [], "center_ks": []},
        "<0.5": {"count": 0, "pred_ks": [], "center_ks": []}
    }
    
    for img_path in sorted_images:
        mask_path = manifest[img_path]
        
        # 检查 harness 是否有对应记录
        if img_path not in harness:
            n_errors += 1
            continue
            
        h_data = harness[img_path]
        w, h = h_data["w"], h_data["h"]
        pred_crop = h_data["crop"]
        keep_area = h_data["keep_area"]
        skipped = h_data["skipped"]
        subject_found = h_data["subject_found"]
        
        # 更新基础统计
        if skipped == 1:
            skipped_count += 1
        if subject_found == 1:
            subject_found_count += 1
        keep_areas.append(keep_area)
        
        # 加载掩膜
        try:
            mask = metrics.load_mask(mask_path)
        except Exception:
            # 读不出掩膜也算 no_mask 或 error？规格说 M 全 False 计入 no_mask。
            # 读不出通常意味着数据损坏，归为 no_mask 或 error。这里归为 no_mask 以便跳过。
            no_mask_count += 1
            continue
            
        if not mask.any():
            no_mask_count += 1
            continue
            
        n_images += 1
        
        # 1. Pred
        kr_pred = metrics.kept_ratio(mask, pred_crop)
        d_pred = metrics.density(mask, pred_crop)
        if kr_pred is not None: pred_ks.append(kr_pred)
        if d_pred is not None: pred_ds.append(d_pred)
        
        # 2. Center
        center_crop_box = metrics.center_crop(w, h, pred_crop[2], pred_crop[3])
        kr_center = metrics.kept_ratio(mask, center_crop_box)
        d_center = metrics.density(mask, center_crop_box)
        if kr_center is not None: center_ks.append(kr_center)
        if d_center is not None: center_ds.append(d_center)
        
        # 3. Optimal
        opt_res = metrics.best_crop_by_mask(mask, pred_crop[2], pred_crop[3])
        opt_crop, opt_sum = opt_res
        if opt_crop is not None:
            kr_optimal = metrics.kept_ratio(mask, opt_crop)
            d_optimal = metrics.density(mask, opt_crop)
            if kr_optimal is not None: optimal_ks.append(kr_optimal)
            if d_optimal is not None: optimal_ds.append(d_optimal)
        else:
            # 无法找到最优框（如窗口大于图），设为 0
            optimal_ks.append(0.0)
            optimal_ds.append(0.0)
            kr_optimal = 0.0
            kr_center_val = kr_center if kr_center is not None else 0.0
            kr_pred_val = kr_pred if kr_pred is not None else 0.0
            
        # 4. Whole
        kr_whole = 1.0
        d_whole = float(mask.mean())
        whole_ks.append(kr_whole)
        whole_ds.append(d_whole)
        
        # Win/Tie/Lose (Pred vs Center)
        # 同时累计幅度：只看命中率会被"赢的次数多但每次都只赢一点点"骗到，
        # 报告里两个数（平均赢多少 / 平均输多少）必须一起看
        kr_p = kr_pred if kr_pred is not None else 0.0
        kr_c = kr_center if kr_center is not None else 0.0
        if kr_p > kr_c:
            wins += 1
            win_gain += kr_p - kr_c
        elif abs(kr_p - kr_c) < 1e-9:
            ties += 1
        else:
            loses += 1
            lose_drop += kr_c - kr_p
            
        # Capture Ratio
        kr_o = kr_optimal if kr_optimal is not None else 0.0
        diff_opt_center = kr_o - kr_c
        if diff_opt_center > 1e-6:
            diff_pred_center = kr_p - kr_c
            cap_num += diff_pred_center
            cap_den += diff_opt_center
            
        # 构图质量：主体在裁剪框里的位置（保留率看不出主体摆得正不正）
        cc_pred = metrics.centroid_in_crop(mask, pred_crop)
        if cc_pred is not None:
            d_thirds = metrics.thirds_distance(cc_pred[0], cc_pred[1])
            d_center = metrics.center_distance(cc_pred[0], cc_pred[1])
            pred_thirds.append(d_thirds)
            pred_center_d.append(d_center)
            if d_center <= DEAD_CENTER_TOL:
                pred_dead += 1
        if kr_pred is not None and kr_pred < CUT_TOL:
            pred_cut += 1

        cc_center = metrics.centroid_in_crop(mask, center_crop_box)
        if cc_center is not None:
            d_thirds = metrics.thirds_distance(cc_center[0], cc_center[1])
            d_center = metrics.center_distance(cc_center[0], cc_center[1])
            center_thirds.append(d_thirds)
            center_center_d.append(d_center)
            if d_center <= DEAD_CENTER_TOL:
                center_dead += 1
        if kr_center is not None and kr_center < CUT_TOL:
            center_cut += 1

        # By Size
        if keep_area >= 0.97:
            b_key = ">=0.97"
        elif keep_area >= 0.8:
            b_key = "0.8~0.97"
        elif keep_area >= 0.5:
            b_key = "0.5~0.8"
        else:
            b_key = "<0.5"
            
        bins[b_key]["count"] += 1
        bins[b_key]["pred_ks"].append(kr_p)
        bins[b_key]["center_ks"].append(kr_c)
        
    # 汇总统计
    def get_mode_stats(ks, ds):
        m_k, md_k = calc_stats(ks)
        m_d, _ = calc_stats(ds)
        return {
            "mean_kept_ratio": round(m_k, 6),
            "median_kept_ratio": round(md_k, 6),
            "mean_density": round(m_d, 6)
        }
        
    def composition_stats(thirds, center_d, dead, cut, n):
        """构图侧的一组汇总数。n_valid 用实际算出来的张数，别用总张数 ——
        主体质心取不到（掩膜为空）的图本来就在上面被跳过了。"""
        n_valid = max(len(thirds), 1)
        return {
            "n": len(thirds),
            "mean_thirds_distance": round(float(np.mean(thirds)) if thirds else 0.0, 6),
            "mean_center_distance": round(float(np.mean(center_d)) if center_d else 0.0, 6),
            "dead_center_rate": round(dead / n_valid, 6),
            "cut_rate": round(cut / max(n, 1), 6),
        }

    def loss_profile(ks):
        """主体损失（被裁掉的主体像素比例）分布。

        输入的 ks 是逐图保留率，None 会被丢掉（无掩膜的图算不出保留率）。
        只统计能算出保留率的图，所以 n 可能小于总张数 —— 这是有意的：
        把"没有主体"的图按损失 0 计进去会把曲线稀释得好看。
        """
        valid = [float(k) for k in ks if k is not None]
        if not valid:
            return {"n": 0}
        arr = np.array(valid)
        lost = 1.0 - arr
        prof = {
            "n": len(valid),
            "mean_lost": round(float(lost.mean()), 6),
            "p90_lost": round(float(np.percentile(lost, 90)), 6),
            "max_lost": round(float(lost.max()), 6),
            "fully_kept_rate": round(float((lost <= 1e-9).mean()), 6),
        }
        for lv in LOSS_LEVELS:
            prof["lost_gt_" + ("%g" % (lv * 100)) + "pct_rate"] = round(float((lost > lv).mean()), 6)
        return prof

    report = {
        "n_images": n_images,
        "n_errors": n_errors,
        "no_mask": no_mask_count,
        "config": {
            "label": args.label,
            "aspect": args.aspect,
            "min_keep": 0.60,
            "manifest": args.manifest,
            "harness": args.harness,
        },
        "pred": get_mode_stats(pred_ks, pred_ds),
        "center": get_mode_stats(center_ks, center_ds),
        "optimal": get_mode_stats(optimal_ks, optimal_ds),
        "whole": get_mode_stats(whole_ks, whole_ds),
        "win_rate": round(wins / max(n_images, 1), 6),
        "tie_rate": round(ties / max(n_images, 1), 6),
        "lose_rate": round(loses / max(n_images, 1), 6),
        "mean_win_gain": round(win_gain / wins, 6) if wins else 0.0,
        "mean_lose_drop": round(lose_drop / loses, 6) if loses else 0.0,
        "capture_ratio": round(cap_num / cap_den, 6) if cap_den > 1e-9 else 0.0,
        "skip_rate": round(skipped_count / max(len(harness), 1), 6),
        "subject_found_rate": round(subject_found_count / max(len(harness), 1), 6),
        "mean_keep_area": round(float(np.mean(keep_areas)), 6) if keep_areas else 0.0,
        "composition": {
            "pred": composition_stats(pred_thirds, pred_center_d, pred_dead, pred_cut, n_images),
            "center": composition_stats(center_thirds, center_center_d, center_dead, center_cut, n_images),
            "dead_center_tol": DEAD_CENTER_TOL,
            "cut_tol": CUT_TOL,
            "loss_levels": list(LOSS_LEVELS),
            "pred_loss": loss_profile(pred_ks),
            "center_loss": loss_profile(center_ks),
            "optimal_loss": loss_profile(optimal_ks),
        },
        "by_size": {},
        "notes": (
            "保留率＝裁剪框内显著性像素数 / 全图显著性像素数（整图恒为 1，越高说明裁掉的废片越少）；"
            "密度＝裁剪框内显著性占比。pred 用 harness 给出的框，center 是同样尺寸的居中框，"
            "optimal 是同样尺寸在所有位置里的上界（所以 pred ≤ optimal 恒成立，capture_ratio 不会超过 1）。"
            "capture_ratio 是提升量之和的比 Σ(pred-center)/Σ(optimal-center)（只在 optimal>center 的图上），"
            "不是逐图比值的平均，避免单张图分母趋 0 时把结果拉飞。"
            "注意 harness 在 64x48 的显著图上搜框，框的宽高比与请求值最大有约 1% 的网格量化偏差，"
            "三方（pred/center/optimal）都在同一套网格上取框，所以对比是公平的。"
        )
    }
    
    # 处理 by_size
    for k, v in bins.items():
        p_m, _ = calc_stats(v["pred_ks"])
        c_m, _ = calc_stats(v["center_ks"])
        report["by_size"][k] = {
            "count": v["count"],
            "pred_mean_kept_ratio": round(p_m, 6),
            "center_mean_kept_ratio": round(c_m, 6)
        }
        
    # 写入 JSON
    out_dir = os.path.dirname(args.out_json)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True)
        
    # 写入 Markdown
    c_pred = report["composition"]["pred"]
    c_center = report["composition"]["center"]
    p_loss = report["composition"]["pred_loss"]
    c_loss = report["composition"]["center_loss"]

    def loss_row(name, prof):
        if not prof.get("n"):
            return f"| {name} | — | — | — | — | — | — | — |"
        cells = " | ".join(
            f"{prof['lost_gt_%gpct_rate' % (lv * 100)] * 100:.1f}%" for lv in LOSS_LEVELS[1:6]
        )
        return (
            f"| {name} | {prof['mean_lost'] * 100:.2f}% | {prof['p90_lost'] * 100:.2f}% | "
            + cells + " |"
        )

    title = "# 构图算法评测报告"
    if args.label:
        title += f"（口径：{args.label}）"
    md_lines = [
        title,
        "",
        f"数据来源：{args.manifest}",
        f"图片数量：{n_images} (有效评测)",
        f"错误行数：{n_errors}",
        f"无掩膜行数：{no_mask_count}",
        "",
        "## 指标概览",
        "",
        "| 模式 | 平均保留率 | 中位保留率 | 平均密度 |",
        "| --- | --- | --- | --- |",
        f"| 算法预测 | {report['pred']['mean_kept_ratio']:.4f} | {report['pred']['median_kept_ratio']:.4f} | {report['pred']['mean_density']:.4f} |",
        f"| 居中裁剪 | {report['center']['mean_kept_ratio']:.4f} | {report['center']['median_kept_ratio']:.4f} | {report['center']['mean_density']:.4f} |",
        f"| 最优位置 | {report['optimal']['mean_kept_ratio']:.4f} | {report['optimal']['median_kept_ratio']:.4f} | {report['optimal']['mean_density']:.4f} |",
        f"| 整图 | {report['whole']['mean_kept_ratio']:.4f} | {report['whole']['median_kept_ratio']:.4f} | {report['whole']['mean_density']:.4f} |",
        "",
        "## 构图质量（主体在框里的位置）",
        "",
        "| 模式 | 主体离三分点 | 主体离框中心 | 摆正中比例 | 切到主体比例 |",
        "| --- | --- | --- | --- | --- |",
        f"| 算法预测 | {c_pred['mean_thirds_distance']:.4f} | {c_pred['mean_center_distance']:.4f} | {c_pred['dead_center_rate']*100:.1f}% | {c_pred['cut_rate']*100:.1f}% |",
        f"| 居中裁剪 | {c_center['mean_thirds_distance']:.4f} | {c_center['mean_center_distance']:.4f} | {c_center['dead_center_rate']*100:.1f}% | {c_center['cut_rate']*100:.1f}% |",
        "",
        "说明：三分点距离＝主体质心到最近那个三分交点的距离（归一化坐标，落在交点为 0，"
        "画面正中心对四个交点都是 ~0.2357 最远）；摆正中＝主体质心离裁剪框中心 5% 以内；"
        "切到主体＝框内留下不到 99% 的主体像素。这一组指标才反映「构图」本身，"
        "保留率只反映「有没有把主体裁掉」。",
        "",
        "## 主体损失分布（被裁掉的主体像素比例）",
        "",
        f"只统计能算出保留率的 {p_loss.get('n', 0)} 张图。损失＝1 − 保留率；"
        ">X% 表示有多大比例的图裁掉了超过 X% 的主体像素。",
        "",
        "| 模式 | 平均损失 | p90 损失 | >1% | >5% | >10% | >20% | >50% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        loss_row("算法预测", p_loss),
        loss_row("居中裁剪", c_loss),
        "",
        "说明：「切到主体比例」只区分 0 与 >0，这一张表才看得出切得有多狠 —— "
        "如果 >1% 有 30% 但 >20% 只有 2%，那说明多出来的那部分是标注噪声级别的小切口，不是真把主体切掉了。",
        "",
        "## 结论",
        "",
        f"算法相对居中裁剪的胜率：{report['win_rate']*100:.1f}%，平局：{report['tie_rate']*100:.1f}%，败率：{report['lose_rate']*100:.1f}%。",
        f"胜负幅度：赢的图平均多留 {report['mean_win_gain']*100:.2f} 个保留率点，"
        f"输的图平均少留 {report['mean_lose_drop']*100:.2f} 个（败率高于胜率但均值仍可能占优，就是靠这个差值）。",
        f"距同样尺寸的最优位置还差 {report['optimal']['mean_kept_ratio'] - report['pred']['mean_kept_ratio']:.4f} 个保留率点"
        f"（最优 {report['optimal']['mean_kept_ratio']:.4f} vs 算法 {report['pred']['mean_kept_ratio']:.4f}）。",
        f"在理论上可提升的空间中（即 optimal 高于 center 的图），算法捕获了 {report['capture_ratio']*100:.1f}% 的提升"
        "（口径：Σ(算法-居中)/Σ(最优-居中)，提升量之和的比，不是逐图比值的平均）。",
        f"算法直接放弃构图优化的比例为 {report['skip_rate']*100:.1f}%（裁剪面积已占整图 97% 以上时不再动框）。",
        f"主体检出率：{report['subject_found_rate']*100:.1f}%。",
        "",
        "说明：harness 在 64x48 的显著性网格上搜框，框的宽高比与请求值最大会有约 1% 的量化偏差；",
        "算法预测、居中基线、最优上界都在同一套网格上取框，因此三方对比是公平的。",
        ""
    ]
    
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    # 打印简要结论
    print(f"评测完成：{n_images} 张有效图片，{n_errors} 个错误。")
    print(f"JSON 报告已保存至：{args.out_json}")
    print(f"Markdown 报告已保存至：{args.out_md}")


if __name__ == "__main__":
    main()
