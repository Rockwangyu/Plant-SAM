clear all;clc;close all;
% 指定要读取图片的文件夹路径
folderPath = 'H:\SAM_test\Rice-Leaf-Disease-Segmentation-Dataset-Code-master\RICE\VOC2012\SegmentationClass'; % 替换为你的文件夹路径

% 获取文件夹中所有图片文件的列表
fileList = dir(fullfile(folderPath, '*.png')); % 假设图片格式为png，可以根据实际情况修改

% 循环处理每张图片
for i = 1:length(fileList)
    % 构建当前图片的完整路径
    currentFile = fullfile(folderPath, fileList(i).name);
    
    % 读取图片
    img = imread(currentFile);
    
    % 将图片转换为八位灰度图像
%     img = rgb2gray(img);
    % 将图片中像素值大于0的部分设置为255
    img(img > 0) = 255;
    % 保存修改后的图片
    [~, name, ext] = fileparts(fileList(i).name);
    outputFileName = fullfile(folderPath, [name ext]);
    imwrite(img, outputFileName);
    
%     % 可选：显示处理后的图片
%     imshow(img);
%     
%     % 可选：等待用户确认查看下一张图片
%     pause;
end
% 
% % % 获取文件夹中所有图片文件的列表
% % fileList = dir(fullfile(folderPath, '*.png')); % 假设图片格式为png，可以根据实际情况修改
% % 
% % % 循环处理每张图片
% % for i = 1:length(fileList)
% %     % 获取当前图片的文件名
% %     currentFile = fullfile(folderPath, fileList(i).name);
% %     
% %     % 构建新的文件名：将 "label" 替换为 "rgb"
% %     [~, name, ext] = fileparts(fileList(i).name);
% %     newName = strrep(name, 'label', 'rgb');
% %     newFileName = [newName ext];
% %     
% %     % 构建新的完整文件路径
% %     newFilePath = fullfile(folderPath, newFileName);
% %     
% %     % 使用 movefile 函数重命名文件
% %     movefile(currentFile, newFilePath);
% % end
% 
% 指定要读取的mat文件路径
% matFilePath = "H:\code_pet\GT_IMG_1 (1).mat"; % 替换为实际的.mat文件路径
% 
% % 使用load函数加载.mat文件
% load(matFilePath);
% 
% % 显示.mat文件中的所有变量
% whos('-file', matFilePath)
% 
% % 例如，假设.mat文件中有一个名为data的变量，打印它的内容
% disp(image_info{1});


