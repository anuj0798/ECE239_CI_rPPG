function [pred_hr] = G(video, frame_rate, video_length, bbox)
% Read in Video and Find Number of Video Frames
input_video = VideoReader(video);
input_video.CurrentTime = 0;
frames=ceil(video_length*input_video.FrameRate); 
% Set up variables needed in face detection, skin segmentation, and green 
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
bvp_vec = rgb_vec(:,2);
nyquist_freq = 1/2*frame_rate; % Nyquist Frequency Formula = 1/2(delta(t))
[b,a] = butter(3,[0.7/nyquist_freq 2.5/nyquist_freq]);% chose 0.7 since general rule for low pass filter should be 70% and 2.5 for high pass filter
bvp_vec = filtfilt(b,a,(double(bvp_vec)-mean(bvp_vec))); % zero phase digital filtering 
bins = 0.5;
i = (60*2*nyquist_freq)/bins;
[pxx,f] = periodogram(bvp_vec,hamming(length(bvp_vec)),i,frame_rate);
limit = (f >= (60/60))&(f <= (100/60));
freq_range = f(limit);
[~,peak] = max(pxx(limit),[],1);
pred = freq_range(peak);
pred_hr = pred*60;
end