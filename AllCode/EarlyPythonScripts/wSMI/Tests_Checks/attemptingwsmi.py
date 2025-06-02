import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from nice.algorithms.connectivity import epochs_compute_wsmi


file_path = 'C:/Users/joshu/PartIIIProject/RSNNdale_attention_1_attention_test'
data = pickle.load(open(file_path, 'rb'))

attend_values = data['attend'].iloc[0]  # Attention labels

# # Validate LFP structure
# print("LFP Type:", type(LFP))
# print("LFP Length:", len(LFP))
# for idx, element in enumerate(LFP):
#     print(f"Element {idx}: Type={type(element)}, Shape={getattr(element, 'shape', None)}")

# Interpret LFP data
left_input = data['LFP'][0][0]  # Left input
right_input = data['LFP'][0][1]  # Right input
attention = data['LFP_rec'][0][2]  # Attention layer
        
plt.plot(left_input[0, :])
plt.show

start_idx = 100
left_input = left_input[:, start_idx:]
right_input = right_input[:, start_idx:]
attention = attention[:, start_idx:]


n_samples = left_input.shape[1]  
n_trials = left_input.shape[0]  


sfreq = 1000  

#separate trials based on attention
attention_left_idx = np.where(attend_values == 0)[0]
attention_right_idx = np.where(attend_values == 1)[0]

#subsets for each attention condition
left_input_left = left_input[attention_left_idx]
right_input_left = right_input[attention_left_idx]
attention_left = attention[attention_left_idx]

left_input_right = left_input[attention_right_idx]
right_input_right = right_input[attention_right_idx]
attention_right = attention[attention_right_idx]

#mne epochs for attention left and attention right
ch_names = ['left_input', 'right_input', 'attention_layer']
ch_types = ['eeg', 'eeg', 'eeg']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

#making RawArray
raw_data_left = np.stack([left_input_left, right_input_left, attention_left], axis=1).reshape(3, -1)
raw_left = mne.io.RawArray(raw_data_left, info)
print("left:", raw_left)

raw_data_right = np.stack([left_input_right, right_input_right, attention_right], axis=1).reshape(3, -1)
raw_right = mne.io.RawArray(raw_data_right, info)
print("right:", raw_right)

#creating events
events_left = np.array([[i * n_samples, 0, 1] for i in range(len(attention_left_idx))])
events_right = np.array([[i * n_samples, 0, 1] for i in range(len(attention_right_idx))])

epochs_left = mne.Epochs(raw_left, events_left, event_id={'Trial': 1}, tmin=0, tmax=(n_samples - 1) / sfreq, baseline=None, preload=True)
epochs_right = mne.Epochs(raw_right, events_right, event_id={'Trial': 1}, tmin=0, tmax=(n_samples - 1) / sfreq, baseline=None, preload=True)

#wSMI for each condition
kernel = 3
taus = [8, 16, 32, 64]  # in ms
wsmi_results = {'left': {}, 'right': {}}

for tau_ms in taus:
    tau_samples = int(tau_ms / (1000 / sfreq))  # Convert ms to samples

    #wSMI for attention left
    wsmi_left, _, _, _ = epochs_compute_wsmi(
        epochs_left, kernel=kernel, tau=tau_samples, backend='python', method_params={'bypass_csd': True}
    )
    wsmi_results['left'][tau_ms] = wsmi_left
    

    #wSMI for attention right
    wsmi_right, _, _, _ = epochs_compute_wsmi(
        epochs_right, kernel=kernel, tau=tau_samples, backend='python', method_params={'bypass_csd': True}
    )
    wsmi_results['right'][tau_ms] = wsmi_right

wsmi_left_input_left = []  # wSMI for left input vs attention layer (attention left)
wsmi_right_input_left = []  # wSMI for right input vs attention layer (attention left)
wsmi_left_input_right = []  # wSMI for left input vs attention layer (attention right)
wsmi_right_input_right = []  # wSMI for right input vs attention layer (attention right)

#average wSMI for each τ for each condition
for tau in taus:
    # For attention left
    wsmi_left_input_left.append(np.mean(wsmi_results['left'][tau][0, 2, :]))  # Left input vs attention layer
    wsmi_right_input_left.append(np.mean(wsmi_results['left'][tau][1, 2, :]))  # Right input vs attention layer

    # For attention right
    wsmi_left_input_right.append(np.mean(wsmi_results['right'][tau][0, 2, :]))  # Left input vs attention layer
    wsmi_right_input_right.append(np.mean(wsmi_results['right'][tau][1, 2, :]))  # Right input vs attention layer


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

#left subplot
axs[0].scatter(taus, wsmi_left_input_left, label="Left Input vs Attention Layer", marker="o", s=100)
axs[0].scatter(taus, wsmi_right_input_left, label="Right Input vs Attention Layer", marker="o", s=100)
axs[0].set_title("Attention Left", fontsize=14)
axs[0].set_xlabel("τ (ms)", fontsize=12)
axs[0].set_ylabel("Average wSMI", fontsize=12)
axs[0].legend()
axs[0].grid(False)

#right subplot
axs[1].scatter(taus, wsmi_left_input_right, label="Left Input vs Attention Layer", marker="o", s=100)
axs[1].scatter(taus, wsmi_right_input_right, label="Right Input vs Attention Layer", marker="o", s=100)
axs[1].set_title("Attention Right", fontsize=14)
axs[1].set_xlabel("τ (ms)", fontsize=12)
axs[1].legend()
axs[1].grid(False)

plt.tight_layout()
plt.show()