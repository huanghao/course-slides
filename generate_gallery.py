import os
import argparse


TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>图片列表</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
        .container { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 15px; justify-content: center; }
        .image-box { text-align: center; }
        .image-box img { width: 100%; max-width: 800px; border-radius: 5px; border: 2px solid #ccc; display: block; }
        .title { font-size: 16px; font-weight: bold; margin: 5px 0; }
    </style>
</head>
<body>
    <h2>图片列表</h2>
    <div class="container">
        {images}
    </div>
</body>
</html>
"""


def generate_gallery(directory, output_html="gallery.html"):
    """生成 HTML 页面，展示目录下所有图片"""
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    images_html = ""

    files = sorted(os.listdir(directory))  # 按文件名排序
    for filename in files:
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(directory, filename)
            images_html += f"""
            <div class="image-box">
                <div class="title">{filename}</div>
                <img src="{file_path}" alt="{filename}">
            </div>
            """

    html_content = TEMPLATE.replace("{images}", images_html)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML 生成完成: {output_html}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_gallery(args.directory)
