# Course Slides 课程幻灯片提取工具

从在线课程视频中提取PPT幻灯片

## 功能特点
- 自动从视频课程中捕获截图
- 识别和过滤幻灯片内容
- 查看剩下PPT图片

## 使用指南

### 1. 视频截图采集
给定一个视频地址，打开selenium，每60秒自动截取一帧视频画面，并按视频标题创建独立目录存储：

```bash
python download_vids.py video_screenshots <视频URL>
```

![自动截图过程](imgs/capture2.png)

### 2. 图片标注
启动Web界面进行标注，区分幻灯片与非幻灯片内容。根据命令行提示，访问 http://localhost:5000 ，开始标注工作。

```bash
cd label-tool
python app.py
```

默认选择负例（非幻灯片：比如老师在讲台上、老师只展示了一半的PPT、大学logo、thank you这种结尾）。不选择默认为正例（需要保留的PPT），代码里可以改一下配置，比如一次同时标20~30张图。由于负例比较少，所以在几个负例上点一下，然后一次批量提交可以标注很多张图片。

![图片标注](imgs/label.jpg)

这是已标注的一些图片样例。5分钟能搞定几百张训练图片。

![标注示例](imgs/samples.jpg)

### 3. 模型训练
使用标注数据训练二分类模型：打开 `classify.ipynb` Jupyter笔记本。按照步骤说明完成模型训练。
用上面标注的这几百张图片，20%验证集，在mac运行2~3分钟，得到一个在验证集上大于90%正确率的模型。

![训练示例](imgs/train2.jpg)

### 4. 幻灯片筛选
使用训练好的模型筛选幻灯片。代码会去重相邻的重复图片（相似度>0.9），然后用上面的二分类判断是否是PPT，只留下PPT。

```bash
# 生成筛选结果
python filter_slides.py <原始图片目录>
```
![filter](imgs/filter.jpg)

### 5. 查看幻灯片
对应展示每张图片是否是PPT，是否是重复。

在看课程的时候，如果有看不懂PPT的内容时，直接右键copy image，然后去问GPT。

![view](imgs/view2.jpg)

### 6. 内容描述生成（TODO）
使用大语言模型为筛选出的幻灯片生成内容描述：

```bash
python generate_descriptions.py <幻灯片目录>
```
