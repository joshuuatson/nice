import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import gc
from scipy.stats import zscore
from scipy.signal import detrend
import copy
import sys
from scipy.integrate import simpson as simps
from nice.algorithms.connectivity import epochs_compute_wsmi
import mne
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import gaussian_filter1d

##this is to check that the loading into epochs works as you would expect
def smooth_with_gaussian(data, sigma=3):
    return gaussian_filter1d(data, sigma=sigma, axis=1) 

def preprocess(data):
    stds = np.std(data[:, :], axis=0)
    non_constant_cols = stds.astype(float) > 1e-6    #finds the time points where std is not 0
    const_cols = stds.astype(float) <= 1e-6    #finds the time points where std is 0

    z = np.zeros_like(data[:, :])   #creates an array of zeros with the same shape as the data
    z[:, non_constant_cols] = zscore(data[:, non_constant_cols], axis=0)  #in the columns where std is not 0, zscores the data
    z[:, const_cols] = 0

 
    if np.isnan(z).any():
        raise ValueError("Data contains NaN values after normalization.")

    return z


file_numbers = [1]

wsmi_means = {f'dataset_{file_number}': {'left_attleft': [], 'right_attleft': [], 'left_attright': [], 'right_attright': []} for file_number in file_numbers}
wsmi_stdevs = {f'dataset_{file_number}': {'left_attleft': [], 'right_attleft': [], 'left_attright': [], 'right_attright': []} for file_number in file_numbers}

total_time = time.time()
for file_number in  file_numbers:
    file_total = time.time()
    file_path = f'C:/Users/joshu/PartIIIProject/RSNNdale_attention_{file_number}_attention_test'
    load_data_start_time = time.time()
    data = pickle.load(open(file_path, 'rb'))
    elapsed_time = time.time() - load_data_start_time
    print(f"Dataset {file_number} loaded in {elapsed_time:.2f} seconds")


    attention_labels = data['label_attend'][0]
    label_left = data['label_left'][0]
    label_right = data['label_right'][0]
    attend_01 = data['attend'][0]
    omitted = data['omit'][0]
    relevant = np.where(omitted ==0)[0]
 
    left_input_SP = data['SP'][0][0][relevant]
    right_input_SP = data['SP'][0][1][relevant]
    attention_SP = data['SP'][0][2][relevant]

    sigma = 2

    left_sm = smooth_with_gaussian(left_input_SP, sigma=sigma) 
    right_sm = smooth_with_gaussian(right_input_SP, sigma=sigma) 
    att_sm = smooth_with_gaussian(attention_SP, sigma=sigma) 

    num_trials, num_samples, num_neurons = left_input_SP.shape
    num_neurons_attention = 80

            
    for j in range(0, num_trials):
        for i in range(0, num_neurons):
            count_left = np.count_nonzero(left_input_SP[j, :, i] == 1)
            if count_left > 0:
                left_sm[j, :, i] /= count_left
            count_right = np.count_nonzero(right_input_SP[j, :, i] == 1)
            if count_right > 0:
                right_sm[j, :, i] /= count_right


        for i in range(0, num_neurons_attention):
            count_attention = np.count_nonzero(attention_SP[j, :, i] == 1)
            if count_attention > 0:
                att_sm[j, :, i] /= count_attention


    left_input_SP = np.sum(left_sm, axis=2)
    right_input_SP = np.sum(right_sm, axis=2)
    attention_SP = np.sum(att_sm, axis=2)

    #preprocess here now that we have traces of all of the relavant trials
    left_indices_agg = np.where((omitted ==0) & (attend_01 == 0))[0]  #indices of agg where left
    _, left_indices, _ = np.intersect1d(relevant, left_indices_agg, return_indices = True)   #indices for relevant processed data where attention left
    right_indices_agg = np.where((omitted ==0) & (attend_01 == 1))[0]
    _, right_indices, _ = np.intersect1d(relevant, right_indices_agg, return_indices = True)

    left_input_SP = preprocess(left_input_SP)
    right_input_SP = preprocess(right_input_SP)
    attention_SP = preprocess(attention_SP)

    #splitting left and right
    left_input_SP_attleft = left_input_SP[left_indices, :]
    right_input_SP_attleft = right_input_SP[left_indices, :]
    attention_SP_attleft = attention_SP[left_indices, :]

    left_input_SP_attright = left_input_SP[right_indices, :]
    right_input_SP_attright = right_input_SP[right_indices, :]
    attention_SP_attright = attention_SP[right_indices, :]


    #----------------------------------------------------------------
    sfreq = 500.0
    ch_names = ['left_input', 'right_input', 'attention_layer']
    ch_types = ['eeg', 'eeg', 'eeg']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


    data_left = np.stack([
        left_input_SP_attleft, 
        right_input_SP_attleft, 
        attention_SP_attleft 
    ], axis=1) 
    print(data_left.shape)

    epochs_left = mne.EpochsArray(
        data_left,
        info, 
        tmin=0, 
        baseline=None
    )
    print("Epoch time axis:")
    print("  first 5 times:", epochs_left.times[:5])
    print("  last 5 times: ", epochs_left.times[-5:])
    # should go from 0.0 up to (n_times-1)/sfreq in steps of 1/sfreq

    data_right = np.stack([
        left_input_SP_attright,
        right_input_SP_attright,
        attention_SP_attright
    ], axis=1) 

    epochs_right = mne.EpochsArray(
        data_right,
        info,
        tmin=0,
        baseline=None
    )
    

    epochs_left.plot(
        n_epochs=5,          # how many trials to show
        n_channels=3,        # number of channels
        scalings='auto',     # autoscale each channel
        title='Attend Left: First 5 Epochs'
    )
    plt.show()


    epochs_right.plot(
        n_epochs=5,
        n_channels=3,
        scalings='auto',
        title='Attend Right: First 5 Epochs'
    )
    plt.show()

  # 2) Recompute the “expected” time axis from your n_times and sfreq:
n_times = data_left.shape[2]         # your original number of samples per epoch
dt     = 1.0 / sfreq
expected_times = np.arange(n_times) * dt
print("Expected time axis matches? ", np.allclose(epochs_left.times, expected_times))

# 3) Spot‑check that the raw epoch data inside MNE is **identical** to your original slice
orig = data_left[0, 0, :]            # trial 0, channel 0, all timepoints
in_ep = epochs_left.get_data()[0, 0, :]
print("Epoch 0, ch0 identical?   ", np.allclose(orig, in_ep))

# 4) If you know there’s, say, a big peak at sample 50 in your original,
#    verify it shows up at t = 50*dt in the epoch plot:
sample_idx = 50
print(f"orig[{sample_idx}] = {orig[sample_idx]:.3f},   epoch[{sample_idx}] = {in_ep[sample_idx]:.3f}")
print(f"time at sample {sample_idx}: {epochs_left.times[sample_idx]:.3f} s")

# 5) (Optional) Quick overlay plot of original vs. epoch for one trial+channel
import matplotlib.pyplot as plt
trial_idxs = [0, 5, 10, 15, 20]
ch = 0  # channel index

fig, axes = plt.subplots(len(trial_idxs), 1, figsize=(8, 2*len(trial_idxs)), sharex=True)

for ax, tr in zip(axes, trial_idxs):
    orig = data_left[tr, ch, :]               # your original stacked array
    in_ep = epochs_left.get_data()[tr, ch, :] # after EpochsArray

    # plot
    ax.plot(expected_times, orig, label=f'orig trial {tr}')
    ax.plot(epochs_left.times, in_ep, '--', label=f'epoch trial {tr}')
    ax.set_ylabel('amplitude')
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()

for tr in range(len(epochs_left)):
    plt.figure(figsize=(6,2))
    plt.plot(expected_times, data_left[tr, 0, :], label='orig')
    plt.plot(epochs_left.times, epochs_left.get_data()[tr, 0, :], '--', label='epoch')
    plt.title(f'Trial {tr}, channel 0')
    plt.legend()
    plt.pause(0.5)   # pause 0.5 s
    plt.clf()
    