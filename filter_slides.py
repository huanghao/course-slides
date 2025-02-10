import os
import argparse
import json

import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import torch

from model import load_model, predict


def is_slide(model, image_path):
    cls, out = predict(model, image_path)
    out = torch.softmax(out, dim=1)
    pred = out[0][1]
    print(cls, pred.item())
    return pred > 0.5


def is_duplicate(img1_path, img2_path, threshold=0.9):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    # **计算 SSIM 相似度**
    similarity = ssim(img1, img2)
    r = similarity > threshold  # 超过 90% 相似度，认为是重复的
    if r:
        print(
            f"SSIM between {os.path.basename(img1_path)} and {os.path.basename(img2_path)}: {similarity:.2f}"
        )
    return r


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("screenshot_folder")
    return p.parse_args()


def main():
    args = parse_args()

    filtered_images = []
    results = {}

    fnames = [
        f for f in os.listdir(args.screenshot_folder) if f.lower().endswith(".png")
    ]
    all_images = sorted(
        fnames, key=lambda fn: int(fn.split("_", 1)[1].split(".", 1)[0])
    )
    if not all_images:
        print("❌ 未找到任何幻灯片截图！")
        return

    model = load_model()

    for image_name in tqdm(all_images):
        image_path = os.path.join(args.screenshot_folder, image_name)

        for i in range(3):
            if len(filtered_images) > i and is_duplicate(
                filtered_images[-i - 1], image_path
            ):
                break
        if i < 2:
            old_image_name = os.path.basename(filtered_images[-i - 1])
            results[old_image_name] = "重复幻灯片"
            results[image_name] = "保留幻灯片"
            filtered_images[-i - 1] = image_path
            print(f"{old_image_name} <- 重复, {image_name} <- 保留")
            continue

        if not is_slide(model, image_path):
            results[image_name] = "非幻灯片"
            print(f"跳过非幻灯片: {image_name}")
            continue

        filtered_images.append(image_path)
        results[image_name] = "保留幻灯片"
        print(f"✔ 保留幻灯片: {image_name}")

    results = sorted(results.items())
    with open(os.path.join(args.screenshot_folder, "filtered_results.json"), "w") as f:
        json.dump(results, f)

    print("✅ 筛选完成！")


if __name__ == "__main__":
    main()
