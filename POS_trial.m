function [RMSE, MAE] = POS_trial(video1, video2, data)
% Read in Video and Find Video Length for front
%video_front = "video_front.mp4";
ground_truth_data_table = data;
video_front = video1;
input_video_f = VideoReader(video_front);
length_video_f = floor(input_video_f.Duration);
frame_rate_f = input_video_f.FrameRate;
frames_f = length_video_f * frame_rate_f;
% Read in Video and Find Video Length for bottom
%video_bottom = "video_bottom.mp4";
video_bottom = video2;
input_video_b = VideoReader(video_bottom);
length_video_b = floor(input_video_b.Duration);
frame_rate_b = input_video_b.FrameRate;
frames_b = length_video_b * frame_rate_b;
% Set up variables needed in face detection, skin segmentation, and POS 
skin_segment = false; 
window = 1.6;
% Front Video
time_vec_f = zeros(frames_f,1);
rgb_vec_f = zeros(frames_f,3);
count_f = 0;
% Bottom Video
time_vec_b = zeros(frames_b,1);
rgb_vec_b = zeros(frames_b,3);
count_b = 0;
% Grab each frame and perform face detection and skin segmentation
% Front Video
while hasFrame(input_video_f) && (input_video_f.CurrentTime <= length_video_f)
    count_f = count_f + 1;
    time_vec_f(count_f) = input_video_f.CurrentTime;
    input_frame_f = readFrame(input_video_f);
    roi_f = input_frame_f;
    if(skin_segment)%skin segmentation - originally specified in reference as an OC-SVM from Wang et al. 2015
        YCBCR = rgb2ycbcr(roi_f);
        Yth = YCBCR(:,:,1)>80;
        CBth = (YCBCR(:,:,2)>77).*(YCBCR(:,:,2)<127);
        CRth = (YCBCR(:,:,3)>133).*(YCBCR(:,:,3)<173);
        roi_skin_f = roi_f.*repmat(uint8(Yth.*CBth.*CRth),[1,1,3]);
        rgb_vec_f(count_f,:) = squeeze(sum(sum(roi_skin_f,1),2)./sum(sum(logical(roi_skin_f),1),2));
    else
        rgb_vec_f(count_f,:) = sum(sum(roi_f,2)) ./ (size(roi_f,1)*size(roi_f,2));
    end   
end
% Bottom Video
while hasFrame(input_video_b) && (input_video_b.CurrentTime <= length_video_b)
    count_b = count_b + 1;
    time_vec_b(count_b) = input_video_b.CurrentTime;
    input_frame_b = readFrame(input_video_b);
    roi_b = input_frame_b;
    if(skin_segment)%skin segmentation - originally specified in reference as an OC-SVM from Wang et al. 2015
        YCBCR = rgb2ycbcr(roi_b);
        Yth = YCBCR(:,:,1)>80;
        CBth = (YCBCR(:,:,2)>77).*(YCBCR(:,:,2)<127);
        CRth = (YCBCR(:,:,3)>133).*(YCBCR(:,:,3)<173);
        roi_skin_b = roi_b.*repmat(uint8(Yth.*CBth.*CRth),[1,1,3]);
        rgb_vec_b(count_b,:) = squeeze(sum(sum(roi_skin_b,1),2)./sum(sum(logical(roi_skin_b),1),2));
    else
        rgb_vec_b(count_b,:) = sum(sum(roi_b,2)) ./ (size(roi_b,1)*size(roi_b,2));
    end   
end
% POS from reference paper pseudocode algorithm
% Front Video
N_f = size(rgb_vec_f,1); % A video sequence containing N frames
H_f = zeros(1,N_f); % Initialize:  H=zeros(1,N)
l_f = ceil(window*frame_rate_f); % l=32 (20 frames/s camera)
for n = 1:N_f-1 % for n=1,2,…,N do
    % C(n)=[R(n),G(n),B(n)]⊤← spatial averaging was performed when video was read
    m = n - l_f + 1; % if m=n−l+1>0 then
    if(m > 0) % if m=n−l+1>0 then
        Cn = ( rgb_vec_f(m:n,:) ./ mean(rgb_vec_f(m:n,:)) )'; % Cin=Cim→nμ(Cim→n)← temporal normalization
        S = [0, 1, -1; -2, 1, 1] * Cn; % S=(0−211−11)⋅Cn← projection
        h = S(1,:) + ((std(S(1,:)) / std(S(2,:))) * S(2,:)); % h=S1+σ(S1)σ(S2)⋅S2← tuning
        H_f(m:n) = H_f(m:n) + (h - mean(h)); % Hm→n=Hm→n+(h−μ(h))← overlap-adding
    end % end if
end % end for
pred_bvp_f = H_f; % Output:The pulse signal H
window_pred_pulse_rate_f = [];
for i = 1:length_video_f
    window_pred_bvp_f = pred_bvp_f(1:frame_rate_f);
    window_pred_pulse_rate_f = prpsd(window_pred_bvp_f,frame_rate_f,60,100);
    window_pred_pulse_rate_f(1,i) = window_pred_pulse_rate_f;
    pred_bvp_f = [pred_bvp_f(frame_rate_f:end)];
end
pred_pulse_rate_f = prpsd(pred_bvp_f,frame_rate_f,60,100);
red_window_pred_pulse_rate_f = window_pred_pulse_rate_f(window_pred_pulse_rate_f~=0); 
avg_window_pred_pulse_rate_f = mean(red_window_pred_pulse_rate_f);
% Bottom Video
N_b = size(rgb_vec_b,1); % A video sequence containing N frames
H_b = zeros(1,N_b); % Initialize:  H=zeros(1,N)
l_b = ceil(window*frame_rate_b); % l=32 (20 frames/s camera)
for n = 1:N_b-1 % for n=1,2,…,N do
    % C(n)=[R(n),G(n),B(n)]⊤← spatial averaging was performed when video was read
    m = n - l_b + 1; % if m=n−l+1>0 then
    if(m > 0) % if m=n−l+1>0 then
        Cn = ( rgb_vec_b(m:n,:) ./ mean(rgb_vec_b(m:n,:)) )'; % Cin=Cim→nμ(Cim→n)← temporal normalization
        S = [0, 1, -1; -2, 1, 1] * Cn; % S=(0−211−11)⋅Cn← projection
        h = S(1,:) + ((std(S(1,:)) / std(S(2,:))) * S(2,:)); % h=S1+σ(S1)σ(S2)⋅S2← tuning
        H_b(m:n) = H_b(m:n) + (h - mean(h)); % Hm→n=Hm→n+(h−μ(h))← overlap-adding
    end % end if
end % end for
pred_bvp_b = H_b; % Output:The pulse signal H
window_pred_pulse_rate_b = [];
for i = 1:length_video_b
    window_pred_bvp_b = pred_bvp_b(1:frame_rate_b);
    window_pred_pulse_rate_b = prpsd(window_pred_bvp_b,frame_rate_b,60,100);
    window_pred_pulse_rate_b(1,i) = window_pred_pulse_rate_b;
    pred_bvp_b = [pred_bvp_b(frame_rate_b:end)];
end
pred_pulse_rate_b = prpsd(pred_bvp_b,frame_rate_b,60,100);
red_window_pred_pulse_rate_b = window_pred_pulse_rate_b(window_pred_pulse_rate_b~=0); 
avg_window_pred_pulse_rate_b = mean(red_window_pred_pulse_rate_b);
% Performance
ground_truth_data = csvread(ground_truth_data_table,1,2);
ground_truth_data_average = mean(ground_truth_data);
error = floor(abs(((pred_pulse_rate_f + pred_pulse_rate_b)/2)- ground_truth_data_average));
mean_error = mean(ground_truth_data - (pred_pulse_rate_f + pred_pulse_rate_b)/2);
RMSE = sqrt(mean_error.^2);
MAE = mean(error);
end

