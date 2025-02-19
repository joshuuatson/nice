import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from nice.algorithms.connectivity import epochs_compute_wsmi
np.set_printoptions(threshold=900)  # Default threshold

# Load data
file_path = 'C:/Users/joshu/PartIIIProject/RSNNdale_attention_1_attention_test'
data = pickle.load(open(file_path, 'rb'))

attention_labels = data['label_attend'][0]
print("attend", attention_labels[:10])
label_left = data['label_left'][0]
print("label_left", label_left[:10])
label_right = data['label_right'][0]
print("label_right", label_right[:10])
attend_01 = data['attend'][0]
print ("attendLR", attend_01[:10])

left_input_LFP = data['LFP'][0][0]  # Left input  [0,1] means left, Right 
right_input_LFP = data['LFP'][0][1]  # Right input
attention_LFP = data['LFP'][0][2]  # Attention layer  [2] means attention 

print(f"Shape of left_input_LFP: {left_input_LFP.shape}, Type: {type(left_input_LFP)}")
print(f"Shape of right_input_LFP: {right_input_LFP.shape}, Type: {type(right_input_LFP)}")
print(f"Shape of attention_LFP: {attention_LFP.shape}, Type: {type(attention_LFP)}")






left_input_avg = np.mean(left_input_LFP, axis=0)
right_input_avg = np.mean(right_input_LFP, axis=0)
attention_avg = np.mean(attention_LFP, axis=0)
plt.plot(left_input_avg, label="Left Input")
plt.plot(right_input_avg, label="Right Input")
plt.plot(attention_avg, label="Attention Layer")
plt.title("LFP ")
plt.xlabel("Time (ms)")
plt.ylabel("Average LFP")
plt.legend()
plt.show()



attention_LFP_in = data['LFP_in'][0][2]   
attention_avg = np.mean(attention_LFP_in, axis=0)
plt.plot(attention_avg, label="Attention Layer")
plt.plot(left_input_avg, label="Left Input")
plt.plot(right_input_avg, label="Right Input")
plt.title("LFP_in")
plt.xlabel("Time (ms)")
plt.ylabel("Average LFP")
plt.legend()
plt.show()


attention_LFP_rec = data['LFP_rec'][0][2]  
attention_avg = np.mean(attention_LFP_rec, axis=0)
plt.plot(attention_avg, label="Attention Layer")
plt.plot(left_input_avg, label="Left Input")
plt.plot(right_input_avg, label="Right Input")
plt.title("LFP_rec")
plt.xlabel("Time (ms)")
plt.ylabel("Average LFP")
plt.legend()
plt.show()

#plotting just the first trials data
sample_idx = 0
plt.plot(left_input_LFP[sample_idx], label="Left Input LFP")
plt.plot(right_input_LFP[sample_idx], label="Right Input LFP")
plt.plot(attention_LFP_rec[sample_idx], label="Attention Layer LFP")
plt.title(f"LFP Data for Trial {sample_idx}")
plt.xlabel("Time (ms)")
plt.ylabel("LFP")
plt.legend()
plt.show()


##similar for the spiking data
left_input_SP = data['SP'][0][0] 
right_input_SP = data['SP'][0][1]
attention_SP = data['SP'][0][2]

print(f"Shape of left_input_SP: {left_input_SP.shape}, Type: {type(left_input_SP)}")
print(f"Shape of right_input_SP: {right_input_SP.shape}, Type: {type(right_input_SP)}")
print(f"Shape of attention_SP: {attention_SP.shape}, Type: {type(attention_SP)}")


reduced_left_input_SP = left_input_SP.mean(axis=2)  # Shape: (2032, 500)
reduced_right_input_SP = right_input_SP.mean(axis=2)
reduced_attention_SP = attention_SP.mean(axis=2)

from scipy.ndimage import gaussian_filter1d


def smooth_with_gaussian(data, sigma=3):
    return gaussian_filter1d(data, sigma=sigma, axis=1) 
smoothed_left_input_SP = smooth_with_gaussian(reduced_left_input_SP, sigma=3)
smoothed_right_input_SP = smooth_with_gaussian(reduced_right_input_SP, sigma=3)
smoothed_attention_SP = smooth_with_gaussian(reduced_attention_SP, sigma=3)


#plotting the smoothed data for just the first sample
sample_idx = 0

plt.plot(smoothed_left_input_SP[sample_idx], label='Left Input SP')
plt.plot(smoothed_right_input_SP[sample_idx], label='Right Input SP)')
plt.plot(smoothed_attention_SP[sample_idx], label='Attention SP')
plt.legend()
plt.title(f"Spike Data for Sample {sample_idx}")
plt.xlabel("Time Steps")
plt.ylabel("Signal")
plt.show()


smoothed_left_mean = np.mean(smoothed_left_input_SP, axis=0)
smoothed_right_mean = np.mean(smoothed_right_input_SP, axis=0)
smoothed_attention_mean = np.mean(smoothed_attention_SP, axis=0)
plt.plot(smoothed_left_mean, label='Left Input SP')
plt.plot(smoothed_right_mean, label='Right Input SP')
plt.plot(smoothed_attention_mean, label='Attention SP')
plt.title("Spike Data Means")
plt.legend()
plt.show()













