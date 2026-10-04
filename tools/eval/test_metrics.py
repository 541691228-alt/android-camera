import unittest
import sys
import os
import math
import tempfile
import numpy as np
from PIL import Image

# 确保当前目录在路径中，以便导入同目录下的 metrics 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import (
    load_mask, crop_slice, kept_ratio, density,
    center_crop, mask_bbox, best_crop_by_mask, iou,
    subject_centroid, centroid_in_crop, thirds_distance, center_distance
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # 准备基础测试数据：10x10 掩膜，中心 3x3 为 True
        self.mask_10x10 = np.zeros((10, 10), dtype=bool)
        self.mask_10x10[3:6, 3:6] = True
        self.total_true = 9

    def test_kept_ratio_exact(self):
        # 包含全部 9 个 True 像素的框，比值应为 1.0
        crop_all = (3, 3, 3, 3)
        self.assertAlmostEqual(kept_ratio(self.mask_10x10, crop_all), 1.0)
        
        # 仅包含 4 个 True 像素的框，比值应为 4/9
        crop_part = (3, 3, 2, 2)
        self.assertAlmostEqual(kept_ratio(self.mask_10x10, crop_part), 4 / 9)

    def test_density_calculation(self):
        # 6x6 框覆盖中心 3x3 区域，密度 = 9 / 36 = 0.25
        crop_6x6 = (2, 2, 6, 6)
        self.assertAlmostEqual(density(self.mask_10x10, crop_6x6), 0.25)

    def test_empty_mask_returns_none(self):
        # 全 False 掩膜，分母为 0 无意义，应返回 None
        empty_mask = np.zeros((10, 10), dtype=bool)
        crop_valid = (0, 0, 5, 5)
        self.assertIsNone(kept_ratio(empty_mask, crop_valid))
        self.assertIsNone(density(empty_mask, crop_valid))

    def test_center_crop_boundaries(self):
        # 奇偶宽度边界测试，确保居中逻辑一致
        # w=11, cw=6 -> (11-6)//2 = 2
        left_odd, top_odd, cw_odd, ch_odd = center_crop(11, 10, 6, 6)
        self.assertEqual(left_odd, 2)
        self.assertEqual(top_odd, 2)
        self.assertEqual((cw_odd, ch_odd), (6, 6))
        # w=10, cw=6 -> (10-6)//2 = 2
        left_even, top_even, _, _ = center_crop(10, 10, 6, 6)
        self.assertEqual(left_even, 2)
        self.assertEqual(top_even, 2)

    def test_best_crop_invariant(self):
        # 随机生成 20 组掩膜，验证最优框保留量 >= 居中框保留量
        rng = np.random.default_rng(0)
        for _ in range(20):
            # 生成 40x30 随机掩膜，阈值 0.8
            mask = rng.random((40, 30)) > 0.8
            cw, ch = 10, 10
            
            # 获取最优框及其总和
            opt_crop, opt_sum = best_crop_by_mask(mask, cw, ch)
            
            # 获取居中框并计算其总和
            center_crop_res = center_crop(30, 40, cw, ch) # w=30, h=40
            y0, y1, x0, x1 = crop_slice(center_crop_res, 40, 30)
            if y0 is None:
                continue
            center_sum = mask[y0:y1, x0:x1].sum()
            
            # 不变式检查：最优解不应差于居中解
            self.assertGreaterEqual(opt_sum, center_sum)

    def test_best_crop_window_too_large(self):
        # 窗口尺寸大于图像尺寸，无法放置，应返回 (None, None)
        mask = np.ones((10, 10), dtype=bool)
        crop, s = best_crop_by_mask(mask, 20, 20)
        self.assertIsNone(crop)
        self.assertIsNone(s)

    def test_crop_slice_clipping(self):
        # 越界裁剪应自动夹取到有效范围
        # (-5, -5, 20, 20) 在 10x10 图上 -> (0, 10, 0, 10)
        result = crop_slice((-5, -5, 20, 20), 10, 10)
        self.assertEqual(result, (0, 10, 0, 10))

    def test_iou_values(self):
        # 相同框 IoU = 1.0
        self.assertEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        
        # 不相交框 IoU = 0.0
        self.assertEqual(iou((0, 0, 5, 5), (10, 10, 5, 5)), 0.0)
        
        # 特定重叠比例 IoU = 0.2
        # 框 A: (0, 0, 1, 3) 面积 3
        # 框 B: (0, 2, 1, 3) 面积 3
        # 重叠: (0, 2, 1, 1) 面积 1
        # 并集: 3 + 3 - 1 = 5
        # IoU: 1/5 = 0.2
        self.assertEqual(iou((0, 0, 1, 3), (0, 2, 1, 3)), 0.2)

    def test_load_mask_roundtrip(self):
        # 使用 Pillow 创建临时灰度 PNG 并验证读取结果
        img = Image.new('L', (10, 10), 0)
        # 绘制部分白色区域 (>=128 为 True)
        for x in range(5):
            for y in range(5):
                img.putpixel((x, y), 255)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img.save(f)
        
        try:
            mask = load_mask(temp_path)
            self.assertEqual(mask.shape, (10, 10))
            self.assertEqual(mask.sum(), 25) # 5x5 区域为 True
        finally:
            os.remove(temp_path)


    def test_subject_centroid(self):
        # 3x3 块在行 3..5 列 3..5，质心取像素中心 -> (4+0.5)/10 = 0.45
        cx, cy = subject_centroid(self.mask_10x10)
        self.assertAlmostEqual(cx, 0.45)
        self.assertAlmostEqual(cy, 0.45)
        # 无主体返回 None，而不是 (0, 0)
        self.assertIsNone(subject_centroid(np.zeros((5, 5), dtype=bool)))

    def test_centroid_in_crop(self):
        # 整图当框：相对坐标等于归一化坐标
        cx, cy = centroid_in_crop(self.mask_10x10, (0, 0, 10, 10))
        self.assertAlmostEqual(cx, 0.45)
        self.assertAlmostEqual(cy, 0.45)
        # 正好框住主体的 3x3 框：质心落在框正中心
        cx2, cy2 = centroid_in_crop(self.mask_10x10, (3, 3, 3, 3))
        self.assertAlmostEqual(cx2, 0.5)
        self.assertAlmostEqual(cy2, 0.5)
        # 主体被完全切到框外时，相对坐标应越界（> 1），而不是被夹回 0..1
        cx3, _ = centroid_in_crop(self.mask_10x10, (0, 0, 3, 3))
        self.assertGreater(cx3, 1.0)

    def test_thirds_and_center_distance(self):
        # 落在三分交点上距离为 0
        self.assertAlmostEqual(thirds_distance(1.0 / 3.0, 1.0 / 3.0), 0.0)
        # 画面正中心最远：到四个交点等距
        self.assertAlmostEqual(thirds_distance(0.5, 0.5), math.hypot(1 / 6, 1 / 6))
        # 中心距离：正中心为 0
        self.assertAlmostEqual(center_distance(0.5, 0.5), 0.0)
        self.assertAlmostEqual(center_distance(0.5, 0.0), 0.5)
        # None 进 None 出，不要抛异常
        self.assertIsNone(thirds_distance(None, 0.5))
        self.assertIsNone(center_distance(None, None))


if __name__ == "__main__":
    unittest.main()
