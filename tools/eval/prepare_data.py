import argparse
import os
import sys
import zipfile
import random
from pathlib import Path


def main():
    # 允许用户通过参数指定路径，方便不同环境复用
    parser = argparse.ArgumentParser(description="准备构图算法评测数据")
    parser.add_argument("--zip", default=os.path.join("data", "DUTS-TE.zip"), help="数据集压缩包路径")
    parser.add_argument("--dest", default="data", help="解压目标目录")
    parser.add_argument("--limit", type=int, default=200, help="抽样数量")
    parser.add_argument("--all", action="store_true", help="不做抽样，全部配对都写进清单（统计功效不够时用）")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--out", default=os.path.join("work", "manifest.tsv"), help="输出清单路径")
    
    args = parser.parse_args()
    
    dest_path = Path(args.dest)
    te_dir = dest_path / "DUTS-TE"
    zip_path = Path(args.zip)
    out_path = Path(args.out)
    
    # 避免重复解压浪费时间和空间
    if not te_dir.exists():
        if not zip_path.exists():
            print(f"错误：找不到压缩包 {zip_path}", file=sys.stderr)
            sys.exit(1)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest_path)
        except Exception as e:
            print(f"错误：解压失败 {e}", file=sys.stderr)
            sys.exit(1)
            
    # 兼容不同命名习惯（如 ImageSet, MASK 等）
    image_dirs = []
    mask_dirs = []
    
    # 递归遍历 dest/DUTS-TE 寻找图像和掩膜目录
    for root, dirs, files in os.walk(te_dir):
        root_name = os.path.basename(root).lower()
        if "image" in root_name:
            image_dirs.append(Path(root))
        elif "mask" in root_name:
            mask_dirs.append(Path(root))
            
    # 建立文件名到绝对路径的映射，便于快速查找
    img_map = {}
    for d in image_dirs:
        for f in d.glob("*.jpg"):
            img_map[f.stem] = str(f.resolve())
            
    mask_map = {}
    for d in mask_dirs:
        for f in d.glob("*.png"):
            mask_map[f.stem] = str(f.resolve())
            
    # 确保图片和掩膜严格对应，防止错位
    pairs = []
    common_stems = sorted(set(img_map.keys()) & set(mask_map.keys()))
    for stem in common_stems:
        pairs.append((img_map[stem], mask_map[stem]))
        
    # 数据量太少无法进行统计学有效的评估
    if not args.all and len(pairs) < args.limit // 2:
        print(f"错误：有效配对数 ({len(pairs)}) 少于限制的一半 ({args.limit // 2})", file=sys.stderr)
        sys.exit(1)
        
    # 固定种子保证结果可复现
    rng = random.Random(args.seed)
    # 防止采样数量超过可用数据导致程序崩溃
    actual_limit = len(pairs) if args.all else min(args.limit, len(pairs))
    sampled_names = common_stems if args.all else rng.sample(common_stems, actual_limit)
    
    # 保证多次运行生成的清单顺序一致
    sampled_names.sort()
    
    # 构建输出列表
    output_lines = ["image\tmask"]
    for name in sampled_names:
        output_lines.append(f"{img_map[name]}\t{mask_map[name]}")
        
    # 避免写入时因父目录不存在而报错
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 强制使用 LF 换行符以符合跨平台规范
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(output_lines))
        
    # 打印关键信息以便确认任务完成状态
    print(f"配对总数：{len(pairs)}")
    print(f"抽样数：{actual_limit}")
    print(f"输出路径：{out_path}")


if __name__ == "__main__":
    main()
