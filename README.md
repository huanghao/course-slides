# course-slides
从公开课视频中提取PPT截图

1）把youtube视频的截图保存在指定的目录下。子目录名称根据视频标题自动创建
> python download_vids.py video_screenshots <some_url>

2）标注正负样例：ppt内容为主的是正例。其他是负例。
> cd label-tool
> python app.py

3）训练一个二分类，判断是不是ppt图片。classify.ipynb

4）通过模型过滤ppt图片。生成filtered_results.json
> python filter_slides.py <origin_dir>

对比展示看过滤后的图片。红色框高亮
> python display_diff_images.py <origin_dir> <filtered_dir>
> open filter_results.html

5）调用大模型对每张ppt生成描述%