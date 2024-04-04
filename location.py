from moviepy.editor import *

video = VideoFileClip('./1.mp4')
print(dir(video))
print(video.size) # 获取分辨率
print(video.duration) # 获取视频总时长
video2 = video.speedx(2)
video2.write_videofile('./3.mp4')