from connectivity import epochs_compute_wsmi
import mne
import numpy as np
import matplotlib.pyplot as plt


# Simulate data: 10 epochs, 64 channels, 1000 time points
n_epochs, n_channels, n_times = 10, 64, 1000
sfreq = 100.0  # Sampling frequency in Hz
data = np.random.randn(n_epochs, n_channels, n_times)

# Create info object
info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types="eeg")

# Create synthetic epochs
epochs = mne.EpochsArray(data, info)


# Parameters
kernel = 3
tau = 2  #use some different values of tau, note that a larger value of tau is picking up lower frequency (aliasing) (64, 32, 16, 8)
tmin = 0
tmax = 0.5  # First half of the epoch
method_params = {"filter_freq": 5.0, "bypass_csd": True}

# Call the function
wsmi, smi, sym, count = epochs_compute_wsmi(
    epochs=epochs,
    kernel=kernel,
    tau=tau,
    tmin=tmin,
    tmax=tmax,
    backend="python",  # Test the Python backend
    method_params=method_params,
    n_jobs=1
)

# print("wSMI:", wsmi)
# print("SMI:", smi)
# print("Symbols:", sym)
# print("Counts:", count)

trial_idx = 0
wsmi_trial = wsmi[:, :, trial_idx]

plt.figure(figsize=(8, 8))
plt.imshow(wsmi_trial, cmap="viridis", aspect="auto")
plt.colorbar(label = "wSMI value")
plt.xlabel("Channel 1")
plt.ylabel("Channel 2")
plt.title("wSMI between all channel pairs for trial {}".format(trial_idx))
plt.show()

# Compute average across trials
wsmi_avg = np.mean(wsmi, axis=2)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(wsmi_avg, cmap="viridis", aspect="auto")
plt.colorbar(label="Average wSMI")
plt.title("Average wSMI Across Trials")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.show()

# Compute average across trials
smi_avg = np.mean(smi, axis=2)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(smi_avg, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Average SMI")
plt.title("Average SMI Across Trials")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.show()

# Flatten the Symbols array and plot a histogram
symbols_flat = sym.flatten()
plt.figure(figsize=(8, 6))
plt.hist(symbols_flat, bins=np.arange(np.min(symbols_flat), np.max(symbols_flat)+1), edgecolor='black')
plt.title("Symbol Frequency Distribution")
plt.xlabel("Symbol")
plt.ylabel("Frequency")
plt.show()


# Select a specific channel and trial
channel_idx = 0
trial_idx = 0
counts_channel_trial = count[channel_idx, :, trial_idx]

# Bar plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(counts_channel_trial)), counts_channel_trial, color='blue', alpha=0.7)
plt.title(f"Symbol Counts for Channel {channel_idx}, Trial {trial_idx}")
plt.xlabel("Symbol")
plt.ylabel("Count")
plt.show()
