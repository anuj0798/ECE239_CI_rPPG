%% POS RUN
%% Data Set 1
% Please change the video files and ground truth data table accordingly
video_front = 'video_front_set1.mp4';
video_bottom = 'video_bottom_set1.mp4';
data1 = 'MPDataExport_set1.csv';
% Front Video Initializations
input_video_f = VideoReader(video_front);
input_video_b = VideoReader(video_bottom);
% Face Detection only find one location since the face does not move much in the video
faceDetector = vision.CascadeObjectDetector();
VidFrame_f = readFrame(input_video_f);
bbox_f = step(faceDetector, VidFrame_f);
a = bbox_f;
VidROI_f = insertShape(VidFrame_f, 'Rectangle', bbox_f);
if numel(bbox_f) > 5 && (bbox_f(1,3) > bbox_f(2,3))
    bbox_f(2,:) = [];
elseif numel(bbox_f) > 5 && (bbox_f(2,3) > bbox_f(1,3))
    bbox_f(1,:) = [];
else
    bbox_f = bbox_f;
end
my_face_f = imcrop(VidROI_f, bbox_f);
figure(1); imshow(my_face_f);
% Bottom Video Initializations
VidFrame_b = readFrame(input_video_b);
bbox_b = step(faceDetector, VidFrame_b);
b = bbox_b
VidROI_b = insertShape(VidFrame_b, 'Rectangle', bbox_b);
if numel(bbox_b) > 5 && (bbox_b(1,3) > bbox_b(2,3))
    bbox_b(2,:) = [];
elseif numel(bbox_b) > 5 && (bbox_b(2,3) > bbox_b(1,3))
    bbox_b(1,:) = [];
else
    bbox_b = bbox_b;
end
my_face_b = imcrop(VidROI_b, bbox_b);
figure(2); imshow(my_face_b);
[H_f_1,H_b_1, MAE1,Rounded_MAE_1, hr_f_1, hr_b_1] = POS_trial(video_front, video_bottom, data1, bbox_f, bbox_b);
%% Data Set 2
% Please change the video files and ground truth data table accordingly
video_front2 = 'video_front.mp4';
video_bottom2 = 'video_bottom.mp4';
data2 = 'MPDataExport.csv';
% Front Video Initializations
input_video_f = VideoReader(video_front2);
input_video_b = VideoReader(video_bottom2);
% Face Detection only find one location since the face does not move much in the video
faceDetector = vision.CascadeObjectDetector();
VidFrame_f = readFrame(input_video_f);
bbox_f = step(faceDetector, VidFrame_f);
a = bbox_f
VidROI_f = insertShape(VidFrame_f, 'Rectangle', bbox_f);
if numel(bbox_f) > 5 && (bbox_f(1,3) > bbox_f(2,3))
    bbox_f(2,:) = [];
elseif numel(bbox_f) > 5 && (bbox_f(2,3) > bbox_f(1,3))
    bbox_f(1,:) = [];
else
    bbox_f = bbox_f;
end
my_face_f = imcrop(VidROI_f, bbox_f);
figure(1); imshow(my_face_f);
% Bottom Video Initializations
VidFrame_b = readFrame(input_video_b);
bbox_b = step(faceDetector, VidFrame_b);
b = bbox_b
VidROI_b = insertShape(VidFrame_b, 'Rectangle', bbox_b);
if numel(bbox_b) > 5 && (bbox_b(1,3) > bbox_b(2,3))
    bbox_b(2,:) = [];
elseif numel(bbox_b) > 5 && (bbox_b(2,3) > bbox_b(1,3))
    bbox_b(1,:) = [];
else
    bbox_b = bbox_b;
end
my_face_b = imcrop(VidROI_b, bbox_b);
figure(2); imshow(my_face_b);
[H_f_2,H_b_2, MAE2,  Rounded_MAE_2, hr_f_2, hr_b_2] = POS_trial(video_front2, video_bottom2, data2, bbox_f, bbox_b);
%% Graphs
X = categorical({'Data Set 1','Data Set 2'});
X = reordercats(X,{'Data Set 1', 'Data Set 2'});
Y = [MAE1, MAE2];
figure(1)
bar(X,Y)
title('Performance of POS on Test Data')
xlabel('Data')
ylabel('MAE')
