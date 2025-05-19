import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from nice.algorithms.connectivity import epochs_compute_wsmi
np.set_printoptions(threshold=900)  # Default threshold

# Load data
file_path = 'C:/Users/joshu/PartIIIProject/RSNNdale_attention_1_attention_test'
data = pickle.load(open(file_path, 'rb'))

#AbsSum = data['LFP'][0]
#this is the cumulative one - no good


left_input = data['LFP'][0][0]  # Left input  [0,1] means left, Right 
right_input = data['LFP'][0][1]  # Right input
attention = data['LFP_rec'][0][2]  # Attention layer  [2] means attention 
#[3] means top-down module. 
#LFP is cumulative sum - but should be the abs sum recurrent and incoming
#LFP_rec is correctly the absolute in and out sum

stimulus = data["label_attend"][0]  # attended stimulus class
stimulus_left = data["label_left"][0]  # stimulus class on left input module
stim_right = data["label_right"][0]  #stimulus class on right input module


attention_labels = data["attend"][0]
attend_left = np.where(data["attend"][0] == 0)[0]  # Trials where attention is on the left
attend_right = np.where(data["attend"][0] == 1)[0]  # Trials where attention is on the right


omitted = data["omit"][0]  # 1: omitted trial, no stimulus shown on the unattended input module (ignore data["label_left"] or data["label_right"] accordingly)
classified_correctly = data["trial_correct"][0]  # 1=trial correctly classified
readout = data["readout_timestamp"][0]  # timestamp at which readout starts (readout duration is always 50 timestamps)
prediction = data["label_predict"][0] # predicted stimulus class

#code that finds the trials when attention was on left or right, but omits the trials where the stimulus was omitted
attend_left_not_omitted = np.where((data["attend"][0] == 0) & (data["omit"][0] == 0))[0]
attend_right_not_omitted = np.where((data["attend"][0] == 1) & (data["omit"][0] == 0))[0]


#so want to plot wsmi of left input and right input, and attention layer
#given we consider only the trials where the stimulus was not omitted


n_samples = left_input.shape[1]  
n_trials = left_input.shape[0]  
print("n_samples:", n_samples)
print("n_trials:", n_trials)

dt = 0.002
sfreq = 1 / dt  # Sampling frequency


#subsets for each attention condition
left_input_attendingleft = left_input[attend_left_not_omitted]
right_input_attendingleft = right_input[attend_left_not_omitted]
attention_layer_attendingleft = attention[attend_left_not_omitted]

left_input_attendingright = left_input[attend_right_not_omitted]
right_input_attendingright = right_input[attend_right_not_omitted]
attention_layer_attendingright = attention[attend_right_not_omitted]
#shapes (495, 500) and (562, 500) respectively

#mne epochs for attention left and attention right
ch_names = ['left_input', 'right_input', 'attention_layer']
ch_types = ['eeg', 'eeg', 'eeg']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

#making RawArray
raw_data_left = np.stack([left_input_attendingleft, right_input_attendingleft, attention_layer_attendingleft], axis=1).reshape(3, -1)
raw_left = mne.io.RawArray(raw_data_left, info)
print("left:", raw_left)

raw_data_right = np.stack([left_input_attendingright, right_input_attendingright, attention_layer_attendingright], axis=1).reshape(3, -1)
raw_right = mne.io.RawArray(raw_data_right, info)
print("right:", raw_right)

#creating events
events_left = np.array([[i * n_samples, 0, 1] for i in range(len(attend_left_not_omitted))])
events_right = np.array([[i * n_samples, 0, 1] for i in range(len(attend_right_not_omitted))])

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

wsmi_left_input_left = []  #wSMI for left input vs attention layer (attention left)
wsmi_right_input_left = []  #wSMI for right input vs attention layer (attention left)
wsmi_left_input_right = []  #wSMI for left input vs attention layer (attention right)
wsmi_right_input_right = []  #wSMI for right input vs attention layer (attention right)

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
axs[0].scatter(taus, wsmi_left_input_left, label="Left Input vs Attention Layer", marker="x",color = 'r', s=100)
axs[0].scatter(taus, wsmi_right_input_left, label="Right Input vs Attention Layer", marker="x",color = 'k', s=100)
axs[0].set_title("Attention Left", fontsize=14)
axs[0].set_xlabel("τ (ms)", fontsize=12)
axs[0].set_ylabel("Average wSMI", fontsize=12)
axs[0].legend()
axs[0].grid(False)

#right subplot
axs[1].scatter(taus, wsmi_left_input_right, label="Left Input vs Attention Layer", marker="x", color = 'r', s=100)
axs[1].scatter(taus, wsmi_right_input_right, label="Right Input vs Attention Layer", marker="x",color = 'k', s=100)
axs[1].set_title("Attention Right", fontsize=14)
axs[1].set_xlabel("τ (ms)", fontsize=12)
axs[1].legend()
axs[1].grid(False)

plt.tight_layout()
plt.show()