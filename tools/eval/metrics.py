import math

import numpy as np
from PIL import Image


def load_mask(path) -> np.ndarray:
    """读 8 位灰度 PNG 掩膜，返回 bool 数组 (H, W)：像素值 >= 128 为 True。
    文件读不出来或不是灰度就按 (H,W) 处理灰度；空数组要抛 ValueError。"""
    img = Image.open(path).convert('L')
    arr = np.array(img)
    if arr.size == 0:
        raise ValueError("Mask array is empty")
    return arr >= 128


def crop_slice(crop, h, w):
    """(left, top, width, height) → (y0, y1, x0, x1)，越界自动夹到 [0,h]/[0,w]，空框返回 None。"""
    if crop is None:
        return None
    l, t, cw, ch = crop
    x0, y0 = l, t
    x1, y1 = l + cw, t + ch
    
    # 坐标夹取到有效范围，防止负索引或超出图像尺寸
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    
    # 检查是否形成有效矩形，宽或高非正数则无效
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def kept_ratio(mask, crop):
    """裁剪框内保留的显著性总量 / 全图显著性总量。
    mask 全 False（没有主体标注）时返回 None（不是 0，因为分母为 0）。"""
    if crop is None:
        return None
    s = crop_slice(crop, mask.shape[0], mask.shape[1])
    if s is None:
        return None
    y0, y1, x0, x1 = s

    total = mask.sum()
    if total == 0:
        return None

    crop_sum = mask[y0:y1, x0:x1].sum()
    return float(crop_sum) / float(total)


def density(mask, crop):
    """裁剪框内 True 的占比（= 裁剪框内 mask.mean()）。空框返回 None。"""
    if crop is None:
        return None
    s = crop_slice(crop, mask.shape[0], mask.shape[1])
    if s is None:
        return None
    y0, y1, x0, x1 = s
    # 整幅图一个主体像素都没有时这个数没有意义（和 kept_ratio 分母为 0 是同一回事），
    # 返回 None 而不是 0.0，免得这个 0 被当成"裁剪框很干净"平均进汇总里
    if not mask.any():
        return None
    return float(mask[y0:y1, x0:x1].mean())


def center_crop(w, h, cw, ch):
    """在 w×h 图里放一个 cw×ch 的居中框，返回 (left, top, cw, ch)。
    left = (w - cw) // 2，top 同理；cw/ch 超过 w/h 时夹到 w/h。"""
    cw = min(cw, w)
    ch = min(ch, h)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return left, top, cw, ch


def mask_bbox(mask):
    """mask 的重心外接矩形 (left, top, width, height)；全 False 返回 None。"""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)


def best_crop_by_mask(mask, cw, ch):
    """在 mask 上滑一个 cw×ch 的窗口，找 True 像素总量最大的位置（积分图，别用暴力双重循环）。
    返回 (crop, best_sum)；图比窗口小或参数非法时返回 (None, None)。"""
    h, w = mask.shape
    if cw > w or ch > h or cw <= 0 or ch <= 0:
        return None, None
    
    # 构建积分图，padding 一圈 0 方便计算
    # I[y+1, x+1] 对应原图 (0,0) 到 (y,x) 的和
    I = np.zeros((h + 1, w + 1), dtype=np.int32)
    I[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int32), axis=1), axis=0)
    
    # 利用切片向量化计算所有可能位置的窗口和
    # 窗口左上角 (y, x) 对应的区域和公式：
    # Sum = I[y+ch, x+cw] - I[y, x+cw] - I[y+ch, x] + I[y, x]
    # 有效 y 范围：0 到 h-ch，有效 x 范围：0 到 w-cw
    
    # 对应 I 的切片索引
    # A: I[ch:, cw:]          -> 右下角
    # B: I[:h-ch+1, cw:]      -> 左下角
    # C: I[ch:, :w-cw+1]      -> 右上角
    # D: I[:h-ch+1, :w-cw+1]  -> 左上角
    
    A = I[ch:, cw:]
    B = I[:h - ch + 1, cw:]
    C = I[ch:, :w - cw + 1]
    D = I[:h - ch + 1, :w - cw + 1]
    
    sums = A - B - C + D
    
    # 找到最大值的位置
    idx = np.unravel_index(np.argmax(sums), sums.shape)
    y_best, x_best = idx
    
    return (int(x_best), int(y_best), int(cw), int(ch)), int(sums[y_best, x_best])


def iou(a, b):
    """两个框的交并比；任一为 None 返回 0.0。"""
    if a is None or b is None:
        return 0.0
    
    l1, t1, w1, h1 = a
    l2, t2, w2, h2 = b
    
    x1 = max(l1, l2)
    y1 = max(t1, t2)
    x2 = min(l1 + w1, l2 + w2)
    y2 = min(t1 + h1, t2 + h2)
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    
    if union == 0:
        return 0.0
    return float(inter_area) / float(union)


def subject_centroid(mask):
    """主体（True 像素）的质心，归一化到 0..1 坐标；没有主体返回 None。
    像素按中心点计（+0.5），免得整幅都是主体时质心贴到右边/下边。"""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    return (float(xs.mean()) + 0.5) / float(w), (float(ys.mean()) + 0.5) / float(h)


def centroid_in_crop(mask, crop):
    """把主体质心换算到裁剪框坐标系：0..1 是框内，越界说明主体被切到框外了。"""
    c = subject_centroid(mask)
    if c is None:
        return None
    s = crop_slice(crop, mask.shape[0], mask.shape[1])
    if s is None:
        return None
    y0, y1, x0, x1 = s
    h, w = mask.shape
    cx = (c[0] * w - x0) / float(x1 - x0)
    cy = (c[1] * h - y0) / float(y1 - y0)
    return cx, cy


# 三分线交点（归一化坐标）
THIRDS_POINTS = (
    (1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 2.0 / 3.0),
)


def thirds_distance(cx, cy):
    """归一化坐标点到最近那个三分交点的欧氏距离。
    画面正中心离四个交点都是 ~0.2357，是最大值；落在交点上为 0。
    这是"构图好不好"的代理指标 —— 保留率只管"有没有把主体裁掉"，管不了"主体摆得正不正"。"""
    if cx is None or cy is None:
        return None
    return float(min(math.hypot(cx - px, cy - py) for px, py in THIRDS_POINTS))


def center_distance(cx, cy):
    """归一化坐标点到画面正中心的距离，用来量化"主体是不是死板地摆中间"。"""
    if cx is None or cy is None:
        return None
    return float(math.hypot(cx - 0.5, cy - 0.5))


if __name__ == "__main__":
    # 构造测试数据：10x10 全 False，中间 3x3 (3:6, 3:6) 为 True
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:6, 3:6] = True
    
    # 1. kept_ratio 测试
    # 包含全部 9 个 True 的框
    r1 = kept_ratio(mask, (3, 3, 3, 3))
    assert abs(r1 - 1.0) < 1e-6, f"kept_ratio full failed: {r1}"
    # 只包含 4 个 True 的框 (3,3,2,2)
    r2 = kept_ratio(mask, (3, 3, 2, 2))
    assert abs(r2 - 4/9) < 1e-6, f"kept_ratio partial failed: {r2}"
    
    # 2. density 测试
    # 6x6 框包含 9 个 True，密度 9/36
    d1 = density(mask, (0, 0, 6, 6))
    assert abs(d1 - 9/36) < 1e-6, f"density failed: {d1}"
    
    # 3. center_crop 奇偶边界
    c1 = center_crop(11, 10, 6, 6)
    assert c1[0] == 2, f"center_crop odd failed: {c1}" # (11-6)//2 = 2
    c2 = center_crop(10, 10, 6, 6)
    assert c2[0] == 2, f"center_crop even failed: {c2}" # (10-6)//2 = 2
    
    # 4. best_crop_by_mask 测试
    best_c, best_s = best_crop_by_mask(mask, 3, 3)
    assert best_c == (3, 3, 3, 3), f"best_crop failed: {best_c}"
    assert best_s == 9, f"best_sum failed: {best_s}"
    
    # 5. iou 测试
    i1 = iou((0, 0, 10, 10), (0, 0, 10, 10))
    assert abs(i1 - 1.0) < 1e-6, f"iou same failed: {i1}"
    i2 = iou((0, 0, 5, 5), (6, 6, 5, 5))
    assert abs(i2 - 0.0) < 1e-6, f"iou disjoint failed: {i2}"
    # 一半重叠：A(0,0,2,2), B(1,0,2,2) -> Inter(1,0,1,2)=2, Area1=4, Area2=4, Union=6 -> 2/6=0.333
    i3 = iou((0, 0, 2, 2), (1, 0, 2, 2))
    expected_i3 = 2.0 / 6.0
    assert abs(i3 - expected_i3) < 1e-6, f"iou overlap failed: {i3}"
    
    # 6. 不变式断言：最优框保留量 >= 居中框保留量
    center_c = center_crop(10, 10, 3, 3)
    center_r = kept_ratio(mask, center_c)
    best_r = kept_ratio(mask, best_c)
    assert best_r >= center_r, f"Invariant failed: best={best_r} >= center={center_r}"
    
    # 7. 构图指标：3x3 块在 x=3..5 行 3..5 列，质心 = (4+0.5)/10 = 0.45
    c = subject_centroid(mask)
    assert abs(c[0] - 0.45) < 1e-9 and abs(c[1] - 0.45) < 1e-9, f"centroid failed: {c}"
    assert subject_centroid(np.zeros((4, 4), dtype=bool)) is None, "empty centroid should be None"
    # 整图框内相对坐标应等于归一化坐标
    cc = centroid_in_crop(mask, (0, 0, 10, 10))
    assert abs(cc[0] - 0.45) < 1e-9 and abs(cc[1] - 0.45) < 1e-9, f"centroid_in_crop failed: {cc}"
    # 框住 3..5 的 3x3 框：质心正好落在框中心
    cc2 = centroid_in_crop(mask, (3, 3, 3, 3))
    assert abs(cc2[0] - 0.5) < 1e-9 and abs(cc2[1] - 0.5) < 1e-9, f"centroid_in_crop tight failed: {cc2}"
    # 三分点距离：落在交点上为 0，画面正中最远（~0.2357）
    assert abs(thirds_distance(1 / 3, 1 / 3)) < 1e-9, "thirds at point should be 0"
    assert abs(thirds_distance(0.5, 0.5) - math.hypot(1 / 6, 1 / 6)) < 1e-9, "thirds at center wrong"
    assert thirds_distance(None, 0.5) is None, "thirds with None should be None"
    assert abs(center_distance(0.5, 0.5)) < 1e-9, "center distance at center should be 0"

    print("metrics self-test OK")
