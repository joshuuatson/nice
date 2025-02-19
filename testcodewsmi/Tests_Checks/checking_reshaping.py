import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from nice.algorithms.connectivity import epochs_compute_wsmi
np.set_printoptions(threshold=100)  # Default threshold

i_values = [1]
for i in i_values:
    # Load data
    file_path = f'C:/Users/joshu/PartIIIProject/RSNNdale_attention_{i}_attention_test'
    data = pickle.load(open(file_path, 'rb'))

    attention_labels = data['label_attend'][0]
    #print("attend", attention_labels[:10])
    label_left = data['label_left'][0]
    #print("label_left", label_left[:10])
    label_right = data['label_right'][0]
    #print("label_right", label_right[:10])
    attend_01 = data['attend'][0]
    # print ("attend01", attend_01[:10])


    #not filtered for omitted trials 
    left_input_LFP = data['LFP'][0][0]  # Left input  [0,1] means left, Right 
    right_input_LFP = data['LFP'][0][1]  # Right input
    attention_LFP = data['LFP_rec'][0][2]  # Attention layer  [2] means attention 
    omitted = data["omit"][0]
    # print("omitted", omitted[:10])
    
    #attend [11. 14. 18. 15.  2.  5. 12. 16.  2.  0.]
    #label_left [11. 14.  6. 18.  2.  5.  6. 10.  3. 14.]
    #label_right [15. 12. 18. 15. 12. 18. 12. 16.  2.  0.]
    #omitted [0. 0. 1. 1. 0. 0. 1. 1. 1. 1.]

    #subset further based on attention
    left_indices = np.where((label_left != label_right) & (omitted == 0) & (attend_01 == 0))[0]
    right_indices = np.where((label_left != label_right) & (omitted == 0) & (attend_01 == 1))[0]

    left_input_LFP_om_left = left_input_LFP[left_indices]
    left_input_LFP_om_left_relevant = left_input_LFP_om_left[:, 100:350]
    #left_input_LFP_om_right_relevant (468, 250)

    print("left_input_LFP_om_left_relevant shape =", left_input_LFP_om_left_relevant.shape)
    #(468, 250)
    print("trial 0=", left_input_LFP_om_left_relevant[0, :])
    print("trial 1=", left_input_LFP_om_left_relevant[1, :])
    print("trial 2=", left_input_LFP_om_left_relevant[2, :])


    right_input_LFP_om_left = right_input_LFP[left_indices]
    right_input_LFP_om_left_relevant = right_input_LFP_om_left[:, 100:350]

    left_input_LFP_om_right = left_input_LFP[right_indices]
    left_input_LFP_om_right_relevant = left_input_LFP_om_right[:, 100:350]

    right_input_LFP_om_right = right_input_LFP[right_indices]
    right_input_LFP_om_right_relevant = right_input_LFP_om_right[:, 100:350]

    attention_LFP_om_left = attention_LFP[left_indices]
    attention_LFP_om_left_relevant = attention_LFP_om_left[:, 100:350]
    attention_LFP_om_right = attention_LFP[right_indices]
    attention_LFP_om_right_relevant = attention_LFP_om_right[:, 100:350]

    print("---for attention left---")
    print("trial 0=", attention_LFP_om_left_relevant[0, :])
    print("trial 1=", attention_LFP_om_left_relevant[1, :])
    print("trial 2=", attention_LFP_om_left_relevant[2, :])


    n_times = left_input_LFP_om_left_relevant.shape[1] ##=250
    print("n_samples for left_input... = ", n_times)
    #250

    dt = 0.002
    sfreq = 1 / dt

    ch_names = ['left_input', 'right_input', 'attention_layer']
    ch_types = ['eeg', 'eeg', 'eeg']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


    #trial 0, left= [2127.54614258 2106.7824707  2157.90454102 ...  242.36921692  251.29391479  264.4413147 ]
    #trial 1, left= [3823.23486328 3863.50585938 3875.29492188 ...  134.72987366  116.69795227  108.72544098]
    #trial 2, left= [3741.21508789 3716.30175781 3689.42480469 ...  187.91952515  184.68182373  162.13943481]

    #trial 0, att= [149.07588196  25.01392746   0.         ...   0.          24.47948837   0.        ]
    #trial 1, att= [122.77959442 154.75906372 137.23712158 ...   0.           0.           0.        ]
    #trial 2, att= [109.21605682 176.77682495 115.0039978  ...   0.          24.47948837  0.        ]




    #reshaping data for attention left
    raw_data_left = np.concatenate([
        left_input_LFP_om_left_relevant, 
        right_input_LFP_om_left_relevant, 
        attention_LFP_om_left_relevant
    ], axis=0)  # Concatenate along time axis

    print("raw_data_left shape =", raw_data_left.shape)  
    #(1404, 250)

    # Reshape into (n_channels, n_samples)
    raw_data_left = raw_data_left.reshape(3, -1) 
    print('raw data left reshaped =', raw_data_left.shape) 
    #3, 117000
    #1404 * 250 / 3 = 117000
    #should check if this it correct reshaping though
    
     
    print('raw data left reshaped, attending left, trial 0:', raw_data_left[0, :n_times])
    print('raw data left reshaped, attending left, trial 1:', raw_data_left[0, n_times:2*n_times])
    print('raw data left reshaped, attending left, trial 2:', raw_data_left[0, 2*n_times:3*n_times])
    #raw data left reshaped, attending left, trial 0: [2127.54614258 2106.7824707  2157.90454102 ...  242.36921692  251.29391479,  264.4413147 ]
    #raw data left reshaped, attending left, trial 1: [3823.23486328 3863.50585938 3875.29492188 ...  134.72987366  116.69795227, 108.72544098]
    #raw data left reshaped, attending left, trial 2: [3741.21508789 3716.30175781 3689.42480469 ...  187.91952515  184.68182373,  162.13943481]

    print('raw data att reshaped, attending left, trial 0:', raw_data_left[2, :n_times])
    print('raw data att reshaped, attending left, trial 1:', raw_data_left[2, n_times:2*n_times])
    print('raw data att reshaped, attending left, trial 2:', raw_data_left[2, 2*n_times:3*n_times])
    #raw data att reshaped, attending left, trial 0: [149.07588196  25.01392746   0.         ...   0.          24.47948837    0.        ]
    #raw data att reshaped, attending left, trial 1: [122.77959442 154.75906372 137.23712158 ...   0.           0.            0.        ]
    #raw data att reshaped, attending left, trial 2: [109.21605682 176.77682495 115.0039978  ...   0.          24.47948837   0.        ]
  

    raw_left = mne.io.RawArray(raw_data_left, info)
    print("raw_data_left =", raw_left)
    



    #reshaping date for attention right 
    raw_data_right = np.concatenate([
        left_input_LFP_om_right_relevant,
        right_input_LFP_om_right_relevant,
        attention_LFP_om_right_relevant
    ], axis=0)

    raw_data_right = raw_data_right.reshape(3, -1)
  
    raw_right = mne.io.RawArray(raw_data_right, info)

    #Creating RawArray with float64 data, n_channels=3, n_times=268000
    #    Range : 0 ... 267999 =      0.000 ...   535.998 secs     why this time? seems to be out by 0.002 seconds of what would be expected

    #defininf event objects, arrays like [0,0,1], [500, 0, 1], [1000, 0, 1] etc
    events_left = np.array([[i * n_times, 0, 1] for i in range(len(left_input_LFP_om_left_relevant))])
    events_right = np.array([[i * n_times, 0, 1] for i in range(len(right_input_LFP_om_right_relevant))])


    print("events_left", events_left[:])
    tmin = 0  # Start of the epoch (relative to the event onset)
    tmax = (250 - 1) / sfreq  # End of the epoch


    epochs_left = mne.Epochs(raw_left, events_left, event_id={'Trial': 1}, tmin=tmin, tmax= tmax, baseline=None, preload=True)
    epochs_right = mne.Epochs(raw_right, events_right, event_id={'Trial': 1}, tmin=tmin, tmax= tmax, baseline=None, preload=True)
   

    print("shape of epochs_left", epochs_left.get_data().shape)
    print(epochs_left.times)
    print("Epoch shape:", epochs_left.get_data().shape)
    print("epochs_left", epochs_left)
    # shape of epochs_left (468, 3, 250)
    #  [0.    0.002 0.004 ... 0.494 0.496 0.498]
    # Epoch shape: (468, 3, 250)
    # epochs_left <Epochs |  468 events (all good), 0 - 0.498 sec, baseline off, ~2.7 MB, data loaded,  'Trial': 468>

    kernel = 3
    taus = [8, 16, 32, 64]  # in ms
    wsmi_results = {'left': {}, 'right': {}}

    for tau in taus:
        tau_samples = int(tau / (1000 / sfreq))
        print(f"tau_samples for {tau}: {tau_samples}")
        
        wsmi_left, _, _, _ = epochs_compute_wsmi(
            epochs_left, kernel=kernel, tau=tau_samples, backend='python', method_params={'bypass_csd': True}
        )
        wsmi_results['left'][tau] = wsmi_left
        #this containts the data for wsmi at a given tau given attending left. 

        wsmi_right, _, _, _ = epochs_compute_wsmi(
            epochs_right, kernel=kernel, tau=tau_samples, backend='python', method_params={'bypass_csd': True}
        )
        wsmi_results['right'][tau] = wsmi_right

    wsmi_left_input_attleft = []  #wSMI for left input vs attention layer (attention left)
    wsmi_right_input_attleft = []  #wSMI for right input vs attention layer (attention left)
    wsmi_left_input_attright = []  #wSMI for left input vs attention layer (attention right)
    wsmi_right_input_attright = []  #wSMI for right input vs attention layer (attention right)

    #average wSMI for each τ for each condition
    for tau in taus:
        # For attention left
        wsmi_left_input_attleft.append(np.mean(wsmi_results['left'][tau][0, 2, :]))  # Left input vs attention layer
        wsmi_right_input_attleft.append(np.mean(wsmi_results['left'][tau][1, 2, :]))  # Right input vs attention layer

        # For attention right
        wsmi_left_input_attright.append(np.mean(wsmi_results['right'][tau][0, 2, :]))  # Left input vs attention layer
        wsmi_right_input_attright.append(np.mean(wsmi_results['right'][tau][1, 2, :]))  # Right input vs attention layer


    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    #left subplot
    axs[0].scatter(taus, wsmi_left_input_attleft, label="Left Input vs Attention Layer", marker="x",color = 'r', s=100)
    axs[0].scatter(taus, wsmi_right_input_attleft, label="Right Input vs Attention Layer", marker="x",color = 'k', s=100)
    axs[0].set_title("Attention Left", fontsize=14)
    axs[0].set_xlabel("τ (ms)", fontsize=12)
    axs[0].set_ylabel("Average wSMI", fontsize=12)
    axs[0].legend()
    axs[0].grid(False)

    #right subplot
    axs[1].scatter(taus, wsmi_left_input_attright, label="Left Input vs Attention Layer", marker="x", color = 'r', s=100)
    axs[1].scatter(taus, wsmi_right_input_attright, label="Right Input vs Attention Layer", marker="x",color = 'k', s=100)
    axs[1].set_title("Attention Right", fontsize=14)
    axs[1].set_xlabel("τ (ms)", fontsize=12)
    axs[1].legend()
    axs[1].grid(False)

    plt.tight_layout()
    plt.show()


    print('wsmi_results.shape', wsmi_results['left'][8].shape)
    #wsmi_results.shape (3, 3, 468), so get a 3x3 for each trial. note that values for performing a single trial are different to 
    #extracting the wsmi value of the first trial from wsmi_results. unsure why. 

 