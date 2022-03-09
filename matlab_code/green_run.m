%% Green Testing
% Data Set 2 Front
video_front = 'video_front.mp4';
input_video_f = VideoReader(video_front);
length_video_f = floor(input_video_f.Duration);
frame_rate_f = input_video_f.FrameRate;
frames_f = length_video_f * frame_rate_f;
[BVP_f, PR_f] = GREEN(video_front, frame_rate_f, 0, length_video_f);
%% Data Set 2 Bottom 
video_bottom = 'video_bottom.mp4';
input_video_b = VideoReader(video_bottom);
length_video_b = floor(input_video_b.Duration);
frame_rate_b = input_video_b.FrameRate;
frames_b = length_video_b * frame_rate_b;
[BVP_b, PR_b] = GREEN(video_bottom, frame_rate_b, 0, length_video_b);