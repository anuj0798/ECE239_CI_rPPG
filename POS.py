import numpy as np
import matplotlib.pyplot as plt 
import cv2

folder_path = '/Users/anuj07/Desktop/UCLA_Q2/239/Project_Videos'
video_path = '/set1/video_front.mp4'
cap = cv2.VideoCapture(folder_path + video_path)


mean_rgb = np.load("/Users/anuj07/Desktop/UCLA_Q2/239/rgb.npy")

winSec = 1.6
FR = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/FR
H = np.zeros(frame_count)
l = int(np.ceil(winSec * FR))#Window length


projection_matrix = np.array([[0,1,-1],[-2,1,1]])
for n in range(frame_count+1):
	m = n - l
	if (m >= 0):
		Cn = (mean_rgb[m:n, :]/ (mean_rgb[m:n, :].mean(axis = 0))).T
		S = projection_matrix @ Cn
		h = S[0, :] + (np.std(S[0, :])/ np.std(S[1, :]))*S[1, :]
		H[m:n] += h -  np.mean(h)



signal = H
plt.plot(range(signal.shape[0]), signal, 'g')
plt.title('Filtered green signal')
plt.show()



# welch
nsegments = 20
# FFT to find the maxiumum frequency
segment_length = (2*signal.shape[0]) // (nsegments + 1) # segment length such that we have 8 50% overlapping segments
print(segment_length)

from scipy.signal import welch
signal = signal.flatten()
green_f, green_psd = welch(signal, FR, 'flattop', nperseg=segment_length) #, scaling='spectrum',nfft=2048)

green_psd = green_psd.flatten()
first = np.where(green_f > 0.8)[0] #0.8 for 300 frames
last = np.where(green_f < 1.8)[0]
first_index = first[0]
last_index = last[-1]
range_of_interest = range(first_index, last_index + 1, 1)

print("Range of interest",range_of_interest)
max_idx = np.argmax(green_psd[range_of_interest])
f_max = green_f[range_of_interest[max_idx]]
hr = f_max*60.0
print("Heart rate = {0}".format(hr))

plt.plot(green_f, green_psd, 'g')
xmax, xmin, ymax, ymin = plt.axis()
plt.vlines(f_max, ymin, ymax, color='red')
plt.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
plt.show()



#periodogram
from scipy.signal import periodogram
Nyquist = FR/2
FResBPM = 0.5 # resolution (bpm) of bins in power spectrum used to determine PR and SNR
N = int(60*2*Nyquist)/FResBPM
f, Pxx_den = periodogram(signal, FR, 'hamming')#,N)

first = np.where(f > 0.8)[0]
last = np.where(f < 1.8)[0]
first_index = first[0]
last_index = last[-1]
range_of_interest = range(first_index, last_index + 1, 1)

max_idx = np.argmax(Pxx_den[range_of_interest])

f_max = f[range_of_interest[max_idx]]
hr = f_max*60.0

plt.plot(f,Pxx_den)
plt.vlines(0.8,0,0.08, color='red')
plt.vlines(1.8,0,0.08, color='red')
plt.vlines(f_max, 0, 0.08, color ='red')
plt.show()

print("Heart rate = {0}".format(hr))
