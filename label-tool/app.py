import os
import random
import json

from flask import Flask, render_template, request, jsonify, send_from_directory

from model import load_model, predict as model_predict
from config import IMAGE_PATH, POSITIVE_LABEL_FILE, NEGATIVE_LABEL_FILE

app = Flask(__name__)

BATCH_SIZE = 12  # 每次显示的图片数量
MODEL_NAME = "slide_classifier_20250222_123401_004"
model = None


@app.route("/predict/<path:image_path>")
def predict(image_path):
    """根据图片路径进行分类"""
    global model
    if model is None:
        model = load_model(MODEL_NAME)

    if not image_path:
        return jsonify({"error": "缺少 image_path 参数"}), 400

    if not os.path.exists(IMAGE_PATH):
        return jsonify({"error": "图片文件不存在"}), 404

    cls, output = model_predict(model, IMAGE_PATH)
    return jsonify(
        {
            "result": "正例" if cls == 1 else "负例",
            "output": output.tolist(),
            "pred": cls,
        }
    )


# **提供静态文件服务**
@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGE_PATH, filename)


def load_labeled_images():
    """读取正例和负例文件，返回两个列表"""
    positive_images, negative_images = set(), set()

    if os.path.exists(POSITIVE_LABEL_FILE):
        with open(POSITIVE_LABEL_FILE, "r") as f:
            positive_images = set(line.strip() for line in f)

    if os.path.exists(NEGATIVE_LABEL_FILE):
        with open(NEGATIVE_LABEL_FILE, "r") as f:
            negative_images = set(line.strip() for line in f)

    return list(positive_images), list(negative_images)


def get_all_images(base_dir):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    images = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                images.append(rel_path)
    return sorted(images)


@app.route("/")
def index():
    all_images = get_all_images(IMAGE_PATH)
    positive_images, negative_images = load_labeled_images()
    labeled_images = set(positive_images + negative_images)
    filtered_images = [img for img in all_images if img not in labeled_images]
    if not filtered_images:
        return jsonify({"msg": "图片全部标注完成"}), 200
    samples = random.sample(filtered_images, min(BATCH_SIZE, len(filtered_images)))
    return render_template(
        "index.html", images=samples, total_unlabeled=len(filtered_images)
    )


@app.route("/submit_batch", methods=["POST"])
def submit_batch():
    data = request.json
    with open(POSITIVE_LABEL_FILE, "a") as pos_f, open(
        NEGATIVE_LABEL_FILE, "a"
    ) as neg_f:
        for image, label in data.items():
            if label == "正":
                pos_f.write(image + "\n")
            else:
                neg_f.write(image + "\n")

    return jsonify({"status": "success", "message": f"已提交 {len(data)} 张图片的标注"})


@app.route("/review")
def review():
    positive_images, negative_images = load_labeled_images()
    positive_images.sort(
        key=lambda x: (
            "_".join(x.split("_")[:-1]),
            int(x.split("_")[-1].split(".", 1)[0]),
        )
    )
    negative_images.sort(
        key=lambda x: (
            "_".join(x.split("_")[:-1]),
            int(x.split("_")[-1].split(".", 1)[0]),
        )
    )
    # import random
    # random.shuffle(positive_images)
    # random.shuffle(negative_images)
    return render_template(
        "review.html", positive_images=positive_images, negative_images=negative_images
    )


@app.route("/select")
def select_directory():
    """显示 BASE_DIR 下的所有子目录"""
    directories = sorted(
        [
            d
            for d in os.listdir(IMAGE_PATH)
            if os.path.isdir(os.path.join(IMAGE_PATH, d))
        ]
    )
    return render_template("select.html", directories=directories)


@app.route("/gallery/<path:subdir>")
def gallery(subdir):
    """展示用户选择的子目录下的图片"""
    target_dir = os.path.join(IMAGE_PATH, subdir)
    if not os.path.exists(target_dir):
        return "目录不存在", 404

    filtered_results = {}
    filtered_results_path = os.path.join(target_dir, "filtered_results.json")
    if os.path.exists(filtered_results_path):
        with open(filtered_results_path, "r") as f:
            filtered_results = dict(json.load(f))

    image_info = []
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    fnames = [
        f
        for f in os.listdir(target_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    for f in sorted(fnames, key=lambda fn: int(fn.split("_", 1)[1].split(".", 1)[0])):
        image_info.append(
            [
                os.path.join(subdir, f),
                f.split("_", 1)[1].split(".", 1)[0],
                filtered_results.get(f, ""),
            ]
        )

    return render_template("gallery.html", image_info=image_info, subdir=subdir)


if __name__ == "__main__":
    app.run(debug=True)
