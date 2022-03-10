function [bvp_vec, pred_pr] = GREEN(video, frame_rate, video_length, bbox)
% Read in Video and Find Number of Video Frames
input_video = VideoReader(video);
input_video.CurrentTime = 0;
frames=ceil(video_length*input_video.FrameRate); 
% Set up variables needed in face detection, skin segmentation, and GREEN 
time_vec = zeros(frames,1);
rgb_vec = zeros(frames,3);
count = 0;
skin_segmentation = true;
while hasFrame(input_video) && (input_video.CurrentTime <= video_length)
    count = count+1;
    time_vec(count) = input_video.CurrentTime;
    input_frame = readFrame(input_video);
    roi_frame = input_frame;
    roi_frame = imcrop(roi_frame, bbox);
    if(skin_segmentation)
        YCBCR = rgb2ycbcr(roi_frame);
        Yth = YCBCR(:,:,1)>80;
        CBth = (YCBCR(:,:,2)>50).*(YCBCR(:,:,2)<130);
        CRth = (YCBCR(:,:,3)>100).*(YCBCR(:,:,3)<180);
        roi_skin = roi_frame.*repmat(uint8(Yth.*CBth.*CRth),[1,1,3]);
        rgb_vec(count,:) = squeeze(sum(sum(roi_skin,1),2)./sum(sum(logical(roi_skin),1),2));
    else
        rgb_vec(count,:) = sum(sum(roi_frame));
end
% Obtain the green channel
bvp_vec = rgb_vec(:,2);
% Use filter and normalize data
nyquist_freq = 1/2*frame_rate; % Nyquist Frequency Formula = 1/2(delta(t))
[b,a] = butter(3,[0.7/nyquist_freq 2.5/nyquist_freq]);%Butterworth 3rd order filter - chose 0.7 since general rule for low pass filter should be 70% and 2.5 for high pass filter
bvp_vec = filtfilt(b,a,(double(bvp_vec)-mean(bvp_vec))); % zero phase digital filtering 
pred_pr = prpsd(bvp_vec,frame_rate,60,100); 
end