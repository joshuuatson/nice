import numpy as np
import pickle
import matplotlib.pyplot as plt
from coherence import field_field_coherence
np.set_printoptions(threshold=100)

i_values = [1, 2]
for i in i_values:
    # Load data
    file_path = f'C:/Users/joshu/PartIIIProject/RSNNdale_attention_{i}_attention_test'
    data = pickle.load(open(file_path, 'rb'))

    attention_labels = data['label_attend'][0]
    label_left = data['label_left'][0]
    label_right = data['label_right'][0]
    attend_01 = data['attend'][0]
    omitted = data["omit"][0]

    # # Subset indices based on omitted and attention conditions:
    # left_indices = np.where((label_left != label_right) & (omitted == 0) & (attend_01 == 0))[0]
    # right_indices = np.where((label_left != label_right) & (omitted == 0) & (attend_01 == 1))[0]

    # Subset indices based on omitted and attention conditions:
    left_indices = np.where((omitted == 0) & (attend_01 == 0))[0]
    right_indices = np.where((omitted == 0) & (attend_01 == 1))[0]

    # Get continuous LFP signals
    left_input_LFP = data['LFP'][0][0]   # Left input
    right_input_LFP = data['LFP'][0][1]  # Right input
    attention_LFP = data['LFP_rec'][0][2]  # Attention layer

    # Subset the LFP arrays for each condition (select trials & relevant time indices)
    left_input_LFP_om_left = left_input_LFP[left_indices][:, 100:350]
    right_input_LFP_om_left = right_input_LFP[left_indices][:, 100:350]
    attention_LFP_om_left = attention_LFP[left_indices][:, 100:350]

    left_input_LFP_om_right = left_input_LFP[right_indices][:, 100:350]
    right_input_LFP_om_right = right_input_LFP[right_indices][:, 100:350]
    attention_LFP_om_right = attention_LFP[right_indices][:, 100:350]

    # Set dt and other parameters
    dt = 0.002  # seconds
    print(f"File {i}: Attention left LFP shape:", attention_LFP_om_left.shape)
    print(f"File {i}: Attention right LFP shape:", attention_LFP_om_right.shape)

    # --- Computing Field-Field Coherence ---
    #for attention left condition:
    left_in_coh_leftatt, freq = field_field_coherence(
        left_input_LFP_om_left,
        attention_LFP_om_left,
        dt
    )
    right_in_coh_leftatt, freq = field_field_coherence(
        right_input_LFP_om_left,
        attention_LFP_om_left,
        dt
    )

    #for attention right condition:
    left_in_coh_rightatt, freq = field_field_coherence(
        left_input_LFP_om_right,
        attention_LFP_om_right,
        dt
    )
    right_in_coh_rightatt, freq = field_field_coherence(
        right_input_LFP_om_right,
        attention_LFP_om_right,
        dt
    )

    # Plotting the coherence results
    plt.figure(figsize=(12, 6))

    # Plot for attention left condition
    plt.subplot(1, 2, 1)
    plt.plot(freq, left_in_coh_leftatt, label='Left Input - Attention Layer', color = 'r')
    plt.plot(freq, right_in_coh_leftatt, label='Right Input - Attention Layer', color = 'k')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title(f'Attention Left Condition - Dataset {i+1}')
    plt.legend()

    # Plot for attention right condition
    plt.subplot(1, 2, 2)
    plt.plot(freq, left_in_coh_rightatt, label='Left Input - Attention Layer', color = 'r')
    plt.plot(freq, right_in_coh_rightatt, label='Right Input - Attention Layer', color = 'k')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title(f'Attention Right Condition - Dataset {i+1}')
    plt.legend()

    plt.tight_layout()
    plt.show()




