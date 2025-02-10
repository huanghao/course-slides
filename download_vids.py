import os
import re
import time
import base64
import argparse
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


@contextmanager
def get_webdriver():
    # 配置 Chrome DevTools
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # 无头模式（可选）
    # chrome_options.add_argument("--disable-gpu")
    # chrome_options.add_argument("--mute-audio")  # 静音
    chrome_options.add_argument("--remote-debugging-port=9222")  # 开启 DevTools

    # 启动浏览器
    driver = webdriver.Chrome(options=chrome_options)
    try:
        yield driver
    finally:
        driver.quit()


def sanitize_filename(filename, replacement="_"):
    """
    将字符串转换为合法的文件/目录名称：
    1. 移除非法字符 (\/:*?"<>|)
    2. 处理空格，替换为 `_`（可选）
    """
    filename = filename.strip()  # 去除首尾空格
    filename = re.sub(r'[\/:*?"<>|]', replacement, filename)  # 替换非法字符
    filename = re.sub(r"\s+", replacement, filename)  # 替换连续空格
    return filename


def save_screenshot(driver, video_url, base_folder):
    driver.get(video_url)
    time.sleep(5)  # 等待页面加载

    # 使用 JavaScript 控制播放
    driver.execute_script(
        """
        let video = document.querySelector('video');
        video.currentTime = 0;  // 从头开始
        video.muted = true;  // 确保静音（防止 YouTube 拦截自动播放）
        video.play();
    """
    )
    time.sleep(5)  # 等待播放启动
    title = driver.title.replace(" - YouTube", "").strip()
    print(f"视频标题: {title}")

    screenshot_folder = os.path.join(base_folder, sanitize_filename(title))
    os.makedirs(screenshot_folder, exist_ok=True)
    print(f"截图保存路径: {screenshot_folder}")

    # **截图并跳过 1 分钟**
    duration = int(
        driver.execute_script("return document.querySelector('video').duration")
    )  # 获取视频时长（秒）
    print(f"视频总时长: {duration:.2f} 秒")
    interval = 60  # 每 60 秒截取一张

    for timestamp in range(0, duration, interval):
        print(f"跳转到 {timestamp} 秒并截图...")
        screenshot_path = os.path.join(
            screenshot_folder, f"screenshot_{timestamp:06d}.png"
        )
        if os.path.exists(screenshot_path):
            continue

        # **跳转到指定时间点**
        driver.execute_script(
            f"document.querySelector('video').currentTime = {timestamp};"
        )
        time.sleep(3)  # 等待视频画面更新

        # **截图**
        # **从 `canvas` 获取 `video` 帧，避免整个网页截图**
        screenshot_base64 = driver.execute_script(
            """
            let video = document.querySelector('video');
            let canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            let ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/png').split(',')[1];
        """
        )
        screenshot_data = base64.b64decode(screenshot_base64)

        # **保存截图**
        with open(screenshot_path, "wb") as f:
            f.write(screenshot_data)

        print(f"截图已保存: {screenshot_path}")

    print("所有截图已完成")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("base_folder")
    p.add_argument("video_url")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.base_folder, exist_ok=True)

    # video_url = "https://www.youtube.com/watch?v=zL9B3eXq0gY&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=30"
    with get_webdriver() as driver:
        save_screenshot(driver, args.video_url, args.base_folder)


if __name__ == "__main__":
    main()
