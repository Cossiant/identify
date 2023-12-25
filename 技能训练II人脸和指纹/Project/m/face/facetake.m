%% 清屏，开始调用摄像头
clc;
clear all;
close all;

%% 定义一个摄像头，并调用，大小是1280*720
vid = videoinput('winvideo', 1,'YUY2_1280x720');

%% 设置摄像头的相关参数(rgb\grayscale)
set(vid,'ReturnedColorSpace','grayscale');
read_video_data = preview(vid);

%% 获取数据帧（视频中的每一帧图像gray_video_data为得到的图像）
figure(Name='image_data');
while ishandle(read_video_data)
    gray_video_data = getsnapshot(vid);

    %% 把灰度图像转化成二值化图像

    BW_video_data = imbinarize(gray_video_data);
    imshow(BW_video_data);

    drawnow
end