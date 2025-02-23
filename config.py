import os

BASE_DIR = "/Users/huanghao/workspace/learning/llm/course-slides"

IMAGE_PATH = os.path.join(BASE_DIR, "video_screenshots")

LABEL_TOOL_PATH = os.path.join(BASE_DIR, "label-tool")

POSITIVE_LABEL_FILE = os.path.join(LABEL_TOOL_PATH, "positive_labels.txt")

NEGATIVE_LABEL_FILE = os.path.join(LABEL_TOOL_PATH, "negative_labels.txt")
