%% Green Implementation
%% Data Set 1 Front
% Please change the video files accordingly
video_front = 'video_front_set1.mp4';
input_video_f = VideoReader(video_front);
length_video_f = floor(input_video_f.Duration);
frame_rate_f = input_video_f.FrameRate;
frames_f = length_video_f * frame_rate_f;
faceDetector = vision.CascadeObjectDetector();
VidFrame = readFrame(input_video_f);
bbox = step(faceDetector, VidFrame);
a = bbox
VidROI = insertShape(VidFrame, 'Rectangle', bbox);
if numel(bbox) > 5 && (bbox(1,3) > bbox(2,3))
    bbox(2,:) = [];
elseif numel(bbox) > 5 && (bbox(2,3) > bbox(1,3))
    bbox(1,:) = [];
else
    bbox = bbox;
end
my_face = imcrop(VidROI, bbox);
figure; imshow(my_face);
[BVP_f1, PR_f1] = GREEN(video_front, frame_rate_f, length_video_f, bbox);
%% Data Set 1 Bottom 
% Please change the video files accordingly
video_bottom = 'video_bottom_set1.mp4';
input_video_b = VideoReader(video_bottom);
length_video_b = floor(input_video_b.Duration);
frame_rate_b = input_video_b.FrameRate;
frames_b = length_video_b * frame_rate_b;
faceDetector = vision.CascadeObjectDetector();
VidFrame = readFrame(input_video_b);
bbox = step(faceDetector, VidFrame);
b = bbox
VidROI = insertShape(VidFrame, 'Rectangle', bbox);
if numel(bbox) > 5 && (bbox(1,3) > bbox(2,3))
    bbox(2,:) = [];
elseif numel(bbox) > 5 && (bbox(2,3) > bbox(1,3))
    bbox(1,:) = [];
else
    bbox = bbox;
end
my_face = imcrop(VidROI, bbox);
figure; imshow(my_face);
[BVP_b1, PR_b1] = GREEN(video_bottom, frame_rate_b,length_video_b, bbox);
%% Data Set 2 Front
% Please change the video files accordingly
video_front = 'video_front.mp4';
input_video_f = VideoReader(video_front);
length_video_f = floor(input_video_f.Duration);
frame_rate_f = input_video_f.FrameRate;
frames_f = length_video_f * frame_rate_f;
faceDetector = vision.CascadeObjectDetector();
VidFrame = readFrame(input_video_f);
bbox = step(faceDetector, VidFrame);
a = bbox
VidROI = insertShape(VidFrame, 'Rectangle', bbox);
if numel(bbox) > 5 && (bbox(1,3) > bbox(2,3))
    bbox(2,:) = [];
elseif numel(bbox) > 5 && (bbox(2,3) > bbox(1,3))
    bbox(1,:) = [];
else
    bbox = bbox;
end
my_face = imcrop(VidROI, bbox);
figure; imshow(my_face);
[BVP_f2, PR_f2] = GREEN(video_front, frame_rate_f,length_video_f, bbox);
%% Data Set 2 Bottom 
% Please change the video files accordingly
video_bottom = 'video_bottom.mp4';
input_video_b = VideoReader(video_bottom);
length_video_b = floor(input_video_b.Duration);
frame_rate_b = input_video_b.FrameRate;
frames_b = length_video_b * frame_rate_b;
faceDetector = vision.CascadeObjectDetector();
VidFrame = readFrame(input_video_b);
bbox = step(faceDetector, VidFrame);
b = bbox
VidROI = insertShape(VidFrame, 'Rectangle', bbox);
if numel(bbox) > 5 && (bbox(1,3) > bbox(2,3))
    bbox(2,:) = [];
elseif numel(bbox) > 5 && (bbox(2,3) > bbox(1,3))
    bbox(1,:) = [];
else
    bbox = bbox;
end
my_face = imcrop(VidROI, bbox);
figure; imshow(my_face);
[BVP_b2, PR_b2] = GREEN(video_bottom, frame_rate_b,  length_video_b, bbox);
%% Performance
X = categorical({'Data Set 1','Data Set 2'});
X = reordercats(X,{'Data Set 1', 'Data Set 2'});
% Please change the ground truth data table accordingly
data1 = 'MPDataExport_set1.csv';
ground_truth_data1 = csvread(data1,1,2);
mean_error1 = mean(ground_truth_data1 - ((PR_b1 + PR_f1)/2));
error1 = (mean(ground_truth_data1) - ((PR_b1+PR_f1)/2));
MAE1_g = abs(mean_error1);
% Please change the ground truth data table accordingly
data2 = 'MPDataExport.csv';
ground_truth_data2 = csvread(data2,1,2);
mean_error2 = mean(ground_truth_data2 - (PR_b2 + PR_f2)/2);
error2 = (mean(ground_truth_data2) - ((PR_b2+PR_f2)/2));
MAE2_g = abs(mean_error2);
Y = [MAE1_g, MAE2_g];
figure(1)
bar(X,Y)
title('Performance of Green on Test Dataset')
xlabel('Data')
ylabel('MAE')


