%% Test POS Implementation
%% Data Set 1
% Please change the video files and ground truth data table file accordingly
video_front = 'video_front_set1.mp4';
video_bottom = 'video_bottom_set1.mp4';
data1 = 'MPDataExport_set1.csv';
% Front Video Initializations
input_video_f = VideoReader(video_front);
input_video_b = VideoReader(video_bottom);
% Face Detection only find one location since the face does not move much in the video
faceDetector = vision.CascadeObjectDetector();
frame_f1 = readFrame(input_video_f);
bbox_f1 = step(faceDetector, frame_f1);
a = bbox_f1;
roi_f1 = insertShape(frame_f1, 'Rectangle', bbox_f1);
if numel(bbox_f1) > 5 && (bbox_f1(1,3) > bbox_f1(2,3))
    bbox_f1(2,:) = [];
elseif numel(bbox_f1) > 5 && (bbox_f1(2,3) > bbox_f1(1,3))
    bbox_f1(1,:) = [];
else
    bbox_f1 = bbox_f1;
end
my_face_f1 = imcrop(roi_f1, bbox_f1);
figure(1); imshow(my_face_f1);
% Bottom Video Initializations
frame_b1 = readFrame(input_video_b);
bbox_b1 = step(faceDetector, frame_b1);
b = bbox_b1;
roi_b1 = insertShape(frame_b1, 'Rectangle', bbox_b1);
if numel(bbox_b1) > 5 && (bbox_b1(1,3) > bbox_b1(2,3))
    bbox_b1(2,:) = [];
elseif numel(bbox_b1) > 5 && (bbox_b1(2,3) > bbox_b1(1,3))
    bbox_b1(1,:) = [];
else
    bbox_b1 = bbox_b1;
end
my_face_b1 = imcrop(roi_b1, bbox_b1);
figure(2); imshow(my_face_b1);
[H_f_1,H_b_1, MAE1,Rounded_MAE_1, hr_f_1, hr_b_1] = POS(video_front, video_bottom, data1, bbox_f1, bbox_b1);
%% Data Set 2
% Please change the video files and ground truth data table file accordingly
video_front2 = 'video_front.mp4';
video_bottom2 = 'video_bottom.mp4';
data2 = 'MPDataExport.csv';
% Front Video Initializations
input_video_f = VideoReader(video_front2);
input_video_b = VideoReader(video_bottom2);
% Face Detection only find one location since the face does not move much in the video
faceDetector = vision.CascadeObjectDetector();
frame_f2 = readFrame(input_video_f);
bbox_f2 = step(faceDetector, frame_f2);
a = bbox_f2;
roi_f2 = insertShape(frame_f2, 'Rectangle', bbox_f2);
if numel(bbox_f2) > 5 && (bbox_f2(1,3) > bbox_f2(2,3))
    bbox_f(2,:) = [];
elseif numel(bbox_f2) > 5 && (bbox_f2(2,3) > bbox_f2(1,3))
    bbox_f2(1,:) = [];
else
    bbox_f2 = bbox_f2;
end
my_face_f2 = imcrop(roi_f2, bbox_f2);
figure(1); imshow(my_face_f2);
% Bottom Video Initializations
frame_b2 = readFrame(input_video_b);
bbox_b2 = step(faceDetector, frame_b2);
b = bbox_b2;
roi_b2 = insertShape(frame_b2, 'Rectangle', bbox_b2);
if numel(bbox_b2) > 5 && (bbox_b2(1,3) > bbox_b2(2,3))
    bbox_b(2,:) = [];
elseif numel(bbox_b2) > 5 && (bbox_b2(2,3) > bbox_b2(1,3))
    bbox_b2(1,:) = [];
else
    bbox_b2 = bbox_b2;
end
my_face_b2 = imcrop(roi_b2, bbox_b2);
figure(2); imshow(my_face_b2);
[H_f_2,H_b_2, MAE2,  Rounded_MAE_2, hr_f_2, hr_b_2] = POS(video_front2, video_bottom2, data2, bbox_f2, bbox_b2);
%% Performance Metric Graphs
X = categorical({'Data Set 1','Data Set 2'});
X = reordercats(X,{'Data Set 1', 'Data Set 2'});
Y = [MAE1, MAE2];
figure(1)
bar(X,Y)
title('Performance of POS on Test Data')
xlabel('Data')
ylabel('MAE')






