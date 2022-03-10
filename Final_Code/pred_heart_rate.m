function [pred_hr] = pred_heart_rate(bvp_vec, frame_rate)
nyquist_freq = 1/2*frame_rate;
bins = 0.5;
i = (60*2*nyquist_freq)/bins;
[pxx,f] = periodogram(bvp_vec,hamming(length(bvp_vec)),i,frame_rate);
limit = (f >= (60/60))&(f <= (100/60));
freq_range = f(limit);
[~,peak] = max(pxx(limit),[],1);
pred = freq_range(peak);
pred_hr = pred*60;
end
