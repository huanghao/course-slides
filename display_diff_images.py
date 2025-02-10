import os
import argparse


def generate_html(original_folder, original_images, filtered_images, output_html):
    html_content = """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>筛选结果展示</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            .container { display: flex; flex-wrap: wrap; justify-content: center; }
            .image-box { margin: 10px; text-align: center; }
            .image-box img { width: 200px; height: auto; border-radius: 5px; }
            .filtered { border: 5px solid red; } /* 筛选出的图片加红框 */
        </style>
    </head>
    <body>
        <h2>筛选结果展示</h2>
        <div class="container">
    """

    for image_name in original_images:
        image_path = os.path.join(original_folder, image_name)
        is_filtered = image_name in filtered_images  # 是否被筛选
        border_class = "filtered" if is_filtered else ""

        html_content += f"""
            <div class="image-box">
                <img src="{image_path}" class="{border_class}">
                <p>{image_name}</p>
            </div>
        """

    # **HTML 结尾**
    html_content += """
        </div>
    </body>
    </html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("screenshot_folder")
    p.add_argument("output_folder")
    return p.parse_args()


def main():
    args = parse_args()

    original_images = sorted(os.listdir(args.screenshot_folder))  # 按名称排序
    filtered_images = set(
        os.listdir(args.output_folder)
    )  # 筛选出的图片（使用 set 加速查找）
    output_html = "filter_results.html"

    generate_html(args.screenshot_folder, original_images, filtered_images, output_html)
    print(f"✅ HTML 生成成功！请用浏览器打开 open {output_html}")


if __name__ == "__main__":
    main()
