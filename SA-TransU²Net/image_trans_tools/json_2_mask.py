import cv2
import json
import numpy as np
import os
import sys
import glob
from tqdm import tqdm


def json_to_binary_mask(json_file: str, output_dir: str) -> None:
    """
    将单个JSON标注文件转换为二值掩码图像

    参数:
    json_file: 输入的JSON标注文件路径
    output_dir: 输出的PNG图像保存目录
    """
    try:
        # 读取JSON文件
        with open(json_file, mode='r', encoding="utf-8") as f:
            configs = json.load(f)

        # 获取图像尺寸
        img_height = configs.get("imageHeight", 0)
        img_width = configs.get("imageWidth", 0)

        if img_height <= 0 or img_width <= 0:
            print(f"错误: {json_file} 中的图像尺寸无效")
            return

        # 创建全黑的二值图像 (0表示背景)
        binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 处理所有标注形状
        shapes = configs.get("shapes", [])
        for shape in shapes:
            # 获取多边形顶点
            points = np.array(shape["points"], np.int32)
            # 在二值图像上填充多边形 (255表示前景)
            cv2.fillPoly(binary_mask, [points], 255)

        # 生成输出文件名
        filename = os.path.splitext(os.path.basename(json_file))[0] + ".png"
        output_path = os.path.join(output_dir, filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 保存二值图像
        cv2.imwrite(output_path, binary_mask)
        print(f"已生成: {output_path}")

    except Exception as e:
        print(f"处理 {json_file} 时出错: {str(e)}")


def batch_process_json_to_mask(input_dir: str, output_dir: str) -> None:
    """
    批量处理目录中的所有JSON文件

    参数:
    input_dir: 输入的JSON文件目录
    output_dir: 输出的PNG图像保存目录
    """
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"警告: 在 {input_dir} 中未找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    # 批量处理每个JSON文件
    for json_file in tqdm(json_files, desc="处理进度"):
        json_to_binary_mask(json_file, output_dir)

    print(f"处理完成! 输出目录: {output_dir}")


if __name__ == "__main__":
    # 设置输入和输出路径，这些路径根据自己的情况设置
    # INPUT_DIR = "D://py_program/data_02/data_json_02/"  # JSON文件所在目录
    # OUTPUT_DIR = "D://py_program/data_02/data_mask/"  # 二值掩码图像输出目录
    INPUT_DIR = r"D:\py_relavant_file\mask"  # JSON文件所在目录（文件夹目录）
    OUTPUT_DIR = r"D:\py_relavant_file\mask"  # 二值掩码图像输出目录（文件夹目录）

    # 检查命令行参数
    if len(sys.argv) > 1:
        INPUT_DIR = sys.argv[1]
        if len(sys.argv) > 2:
            OUTPUT_DIR = sys.argv[2]

    # 确保输入目录存在
    if not os.path.isdir(INPUT_DIR):
        print(f"错误: 输入目录不存在 - {INPUT_DIR}")
        sys.exit(1)

    # 执行批量处理
    batch_process_json_to_mask(INPUT_DIR, OUTPUT_DIR)
