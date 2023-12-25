clc;
clear;
close all;
%% 使用前请调用一次
% pyversion(".\Python308\python.exe");
% 这是相关使用教程
% https://blog.csdn.net/wenhao_ir/article/details/124888473

img = py.cv2.imread(".\Image\face\test\113676980_p0.png");

py.cv2.imshow("img",img);

py.cv2.waitKey();