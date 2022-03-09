function [BVP, PR] = GREEN(VideoFile, FS, StartTime, Duration)

%Parameters
LPF = 0.7; %low cutoff frequency (Hz) - 0.8 Hz in reference
HPF = 2.5; %high cutoff frequency (Hz) - both 6.0 Hz and 2.0 Hz used in reference


%Load Video:
VidObj = VideoReader(VideoFile);
VidObj.CurrentTime = StartTime;

FramesToRead=ceil(Duration*VidObj.FrameRate); %video may be encoded at slightly different frame rate

% Read Video and Spatially Average:
T = zeros(FramesToRead,1);%initialize time vector
RGB = zeros(FramesToRead,3);%initialize color signal
FN = 0;
while hasFrame(VidObj) && (VidObj.CurrentTime <= StartTime+Duration)
    FN = FN+1;
    T(FN) = VidObj.CurrentTime;
    VidFrame = readFrame(VidObj);
    
    %position for optional face detection/tracking - originally specified in reference as a manual segmentation.
    VidROI = VidFrame;
    
    %position for optional skin segmentation
    
    RGB(FN,:) = sum(sum(VidROI));%if different size regions are used for different frames, the signals should be normalized by the region size, but not necessary for whole frame processing or constant region size
end

% Select BVP Source:
% Green channel
BVP = RGB(:,2);

% Filter, Normalize
%NyquistF = 1/2*FS;
%[B,A] = butter(3,[LPF/NyquistF HPF/NyquistF]);%Butterworth 3rd order filter - originally specified in reference with a 4th order butterworth using filtfilt function
%BVP_F = filtfilt(B,A,(double(BVP)-mean(BVP)));
%
%BVP = BVP_F;

% Estimate Pulse Rate from periodogram
PR = prpsd(BVP,FS,60,100);
end