import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import gc
from coherence import field_field_coherence
from scipy.stats import zscore
from scipy.signal import detrend
import copy

run_time = time.time()
file_numbers = [1, 2, 4, 8, 14, 15, 20, 23]

classes_left = np.arange(0, 20)
classes_right = np.arange(0, 20)

agg_structure = {f'dataset_{file_number}': 
                 {f'left_class_{class_left}': 
                  {f'right_class_{class_right}': [] for class_right in classes_right} for class_left in classes_left} for file_number in file_numbers}

left_input_LFP_attleft_agg = copy.deepcopy(agg_structure)
right_input_LFP_attleft_agg = copy.deepcopy(agg_structure)
attention_LFP_attleft_agg = copy.deepcopy(agg_structure)

left_input_LFP_attright_agg = copy.deepcopy(agg_structure)
right_input_LFP_attright_agg = copy.deepcopy(agg_structure)
attention_LFP_attright_agg = copy.deepcopy(agg_structure)

n_values = {f'dataset_{file_number}':
            {f'left_class_{class_left}':
             {f'right_class_{class_right}': {
                'attleft': [],
                'attright': []
                } for class_right in classes_right} for class_left in classes_left} for file_number in file_numbers}

#loading the data by class combinations
total_time = time.time()
total_load_time = time.time()
for file_number in file_numbers:   #will first load the file and extract the data
    file_path = f'C:/Users/joshu/PartIIIProject/RSNNdale_attention_{file_number}_attention_test'
    load_data_start_time = time.time()
    data = pickle.load(open(file_path, 'rb'))
    elapsed_time = time.time() - load_data_start_time
    print(f"Dataset {file_number} loaded in {elapsed_time:.2f} seconds")
    file_process_time = time.time()

    attention_labels = data['label_attend'][0]
    label_left = data['label_left'][0]
    label_right = data['label_right'][0]
    attend_01 = data['attend'][0]
    omitted = data["omit"][0]

    # Extract LFP signals
    left_input_LFP = data['LFP'][0][0]
    right_input_LFP = data['LFP'][0][1]
    attention_LFP = data['LFP'][0][2]

    for class_left in classes_left:
         for class_right in classes_right: 
    
            left_indices = np.where((omitted ==0) & (attend_01 == 0) & (label_left == class_left) & (label_right == class_right))[0]
            right_indices = np.where((omitted ==0) & (attend_01 == 1) & (label_left == class_left) & (label_right == class_right))[0]

            n_values[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'][f'attleft'] = [len(left_indices)]  #e.g. checks number of trials of this class combo for attentino left
            n_values[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'][f'attright'] = [len(right_indices)]  #nb need both left and right indices to be non empty

            if len(left_indices) >= 1:
                left_input_LFP_attleft = left_input_LFP[left_indices][:, 100:350]
                right_input_LFP_attleft = right_input_LFP[left_indices][:, 100:350]
                attention_LFP_attleft = attention_LFP[left_indices][:, 100:350]

                left_input_LFP_attleft_agg[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(left_input_LFP_attleft)
                right_input_LFP_attleft_agg[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(right_input_LFP_attleft)
                attention_LFP_attleft_agg[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(attention_LFP_attleft)

            if len(right_indices) >= 1:

                left_input_LFP_attright = left_input_LFP[right_indices][:, 100:350]
                right_input_LFP_attright = right_input_LFP[right_indices][:, 100:350]
                attention_LFP_attright = attention_LFP[right_indices][:, 100:350]

                left_input_LFP_attright_agg[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(left_input_LFP_attright)   
                right_input_LFP_attright_agg[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(right_input_LFP_attright)
                attention_LFP_attright_agg[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(attention_LFP_attright)

    del data
    gc.collect()
    elapsed_time = time.time() - file_process_time
    print(f"Dataset {file_number} processed in {elapsed_time:.2f} seconds")


#dont want to process here - want to process when its in the big array
print(f'Total load time = {time.time() - total_load_time:.2f} seconds')

#-------formatting the trials into a single array for each class pair, of length n_samples * n_trials------------------
all_structure = {f'left_class_{class_left}': {f'right_class_{class_right}': [] for class_right in classes_right} for class_left in classes_left}

left_input_LFP_attleft_all = copy.deepcopy(all_structure)
right_input_LFP_attleft_all = copy.deepcopy(all_structure)
attention_LFP_attleft_all = copy.deepcopy(all_structure)

left_input_LFP_attright_all = copy.deepcopy(all_structure)
right_input_LFP_attright_all = copy.deepcopy(all_structure)
attention_LFP_attright_all = copy.deepcopy(all_structure)

for filenumber in file_numbers:
    for class_left in classes_left:
        for class_right in classes_right:
            left_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'].extend(left_input_LFP_attleft_agg[f'dataset_{filenumber}'][f'left_class_{class_left}'][f'right_class_{class_right}'])
            right_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'].extend(right_input_LFP_attleft_agg[f'dataset_{filenumber}'][f'left_class_{class_left}'][f'right_class_{class_right}'])
            attention_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'].extend(attention_LFP_attleft_agg[f'dataset_{filenumber}'][f'left_class_{class_left}'][f'right_class_{class_right}'])

            left_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'].extend(left_input_LFP_attright_agg[f'dataset_{filenumber}'][f'left_class_{class_left}'][f'right_class_{class_right}'])
            right_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'].extend(right_input_LFP_attright_agg[f'dataset_{filenumber}'][f'left_class_{class_left}'][f'right_class_{class_right}'])
            attention_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'].extend(attention_LFP_attright_agg[f'dataset_{filenumber}'][f'left_class_{class_left}'][f'right_class_{class_right}'])

#------ want to preprocess each array here now that they are properly compiled ----------------
#looking first at the shape / structure of left_input_LFP_attleft_all:
print('left_input_LFP_attleft_all 00 raw =', left_input_LFP_attleft_all[f'left_class_{0}'][f'right_class_{0}'])
print('left_input_LFP_attleft_all 01 raw =', left_input_LFP_attleft_all[f'left_class_{0}'][f'right_class_{1}'])

for class_left in classes_left:
        for class_right in classes_right:
            left_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.vstack(left_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
            right_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.vstack(right_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
            attention_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.vstack(attention_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'])

            left_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.vstack(left_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
            right_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.vstack(right_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
            attention_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.vstack(attention_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'])

print('left_input_LFP_attleft_all 00 reshaped = ', left_input_LFP_attleft_all[f'left_class_{0}'][f'right_class_{0}'])
print('left_input_LFP_attleft_all 01 reshaped = ', left_input_LFP_attleft_all[f'left_class_{0}'][f'right_class_{1}'])

def preprocess_data(data):
    # data = detrend(data, axis = 0)   #data seems to look nicer if i do this

    for i in range(len(data)):
        data[i] = detrend(data[i])

    data = zscore(data, axis = 0)
    return data

left_attleft = copy.deepcopy(all_structure)
right_attleft = copy.deepcopy(all_structure)
att_attleft = copy.deepcopy(all_structure)

left_attright = copy.deepcopy(all_structure)
right_attright = copy.deepcopy(all_structure)
att_attright = copy.deepcopy(all_structure)

for class_left in classes_left:
    for class_right in classes_right:
        left_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'] = preprocess_data(left_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
        right_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'] = preprocess_data(right_input_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
        att_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'] = preprocess_data(attention_LFP_attleft_all[f'left_class_{class_left}'][f'right_class_{class_right}'])

        left_attright[f'left_class_{class_left}'][f'right_class_{class_right}'] = preprocess_data(left_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
        right_attright[f'left_class_{class_left}'][f'right_class_{class_right}'] = preprocess_data(right_input_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'])
        att_attright[f'left_class_{class_left}'][f'right_class_{class_right}'] = preprocess_data(attention_LFP_attright_all[f'left_class_{class_left}'][f'right_class_{class_right}'])  

print('left_attleft 00 =', left_attleft[f'left_class_{0}'][f'right_class_{0}'])
print('left_attleft 01 =', left_attleft[f'left_class_{0}'][f'right_class_{1}'])

   
n_values_sum = {f'left_class_{class_left}': {f'right_class_{class_right}': {'attleft': 0, 'attright': 0} for class_right in classes_right} for class_left in classes_left}


for class_left in classes_left:
    for class_right in classes_right:
        n_values_sum[f'left_class_{class_left}'][f'right_class_{class_right}']['attleft'] = sum(
            n_values[f'dataset_{dataset}'][f'left_class_{class_left}'][f'right_class_{class_right}']['attleft'][0] for dataset in file_numbers  #this will check through all the datasets 
        )
        n_values_sum[f'left_class_{class_left}'][f'right_class_{class_right}']['attright'] = sum(
            n_values[f'dataset_{dataset}'][f'left_class_{class_left}'][f'right_class_{class_right}']['attright'][0] for dataset in file_numbers
        )

#-------------wsmi calculation----------------
dt = 0.002
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
freq_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 80)]  # Actual frequency ranges


lr_split_structure = {f'left_class_{class_left}': {f'right_class_{class_right}': {f'{band}':  {
    'left_attleft': [], 'right_attleft': [], 'left_attright': [], 'right_attright': []
    } for band in bands} for class_right in classes_right} for class_left in classes_left} 


coh_means = copy.deepcopy(lr_split_structure)
coh_stdevs = copy.deepcopy(lr_split_structure)


for class_left in classes_left:
    coh_start = time.time()
    for class_right in classes_right: 

        if n_values_sum[f'left_class_{class_left}'][f'right_class_{class_right}']['attleft'] == 0:  ##this makes sure you only calulate for left class pairs that have trials
            continue                    

        left_in_coh_leftatt, freq = field_field_coherence(
            left_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'],
            att_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'],
            dt
        )
        right_in_coh_leftatt, freq = field_field_coherence(
            right_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'],
            att_attleft[f'left_class_{class_left}'][f'right_class_{class_right}'],
            dt
        )
          
        for band, (f_min, f_max) in zip(bands, freq_ranges):
            band_idx = (freq >= f_min) & (freq < f_max)
            coh_means[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['left_attleft'].append(
                np.mean(left_in_coh_leftatt[band_idx]))  #taking the mean across trials
            coh_means[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['right_attleft'].append(
                np.mean(right_in_coh_leftatt[band_idx]))

            coh_stdevs[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['left_attleft'].append(
                np.std(left_in_coh_leftatt[band_idx], ddof=1))
            coh_stdevs[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['right_attleft'].append(
                np.std(right_in_coh_leftatt[band_idx], ddof=1))

    print(f"Class pair {class_left} {class_right} processed in {time.time() - coh_start:.2f} seconds")



for class_left in classes_left:
    coh_start = time.time()
    for class_right in classes_right:

        if n_values_sum[f'left_class_{class_left}'][f'right_class_{class_right}']['attright'] == 0:  #right class pairs that have trials
            continue        

        #for attention right condition:
        left_in_coh_rightatt, freq = field_field_coherence(
            left_attright[f'left_class_{class_left}'][f'right_class_{class_right}'],
            att_attright[f'left_class_{class_left}'][f'right_class_{class_right}'],
            dt
        )
        right_in_coh_rightatt, freq = field_field_coherence(
            right_attright[f'left_class_{class_left}'][f'right_class_{class_right}'],
            att_attright[f'left_class_{class_left}'][f'right_class_{class_right}'],
            dt
        )   

        for band, (f_min, f_max) in zip(bands, freq_ranges):
            band_idx = (freq >= f_min) & (freq < f_max)
            coh_means[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['left_attright'].append(
                np.mean(left_in_coh_rightatt[band_idx]))  #taking the mean across trials
            coh_means[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['right_attright'].append(
                np.mean(right_in_coh_rightatt[band_idx]))

            coh_stdevs[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['left_attright'].append(
                np.std(left_in_coh_rightatt[band_idx], ddof=1))
            coh_stdevs[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['right_attright'].append(
                np.std(right_in_coh_rightatt[band_idx], ddof=1))


    print(f"Class pair {class_left} {class_right} processed in {time.time() - coh_start:.2f} seconds")
         
run_time = time.time() - run_time
print(f"Total run time = {run_time:.2f} seconds")



#--------plotting the results-------------
results_to_plot =copy.deepcopy(lr_split_structure) 

for band in bands:
    for class_left in classes_left:
        for class_right in classes_right:
            for key in ['left_attleft', 'right_attleft', 'left_attright', 'right_attright']:
                results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'][band][key] = coh_means[f'left_class_{class_left}'][f'right_class_{class_right}'][band][key]


def calculate_vmin_vmax(results_to_plot, classes_left, classes_right, band):
    vmax = np.max([results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'][band][key] 
                   for class_left in classes_left for class_right in classes_right for key in ['left_attleft', 'right_attleft', 'left_attright', 'right_attright']])
    vmin = np.min([results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'][band][key] 
                   for class_left in classes_left for class_right in classes_right for key in ['left_attleft', 'right_attleft', 'left_attright', 'right_attright']])
    return vmin, vmax

print(results_to_plot[f'left_class_{0}'][f'right_class_{0}']['delta']['left_attleft'])

vmin_delta, vmax_delta = calculate_vmin_vmax(results_to_plot, classes_left, classes_right, 'delta')
vmin_theta, vmax_theta = calculate_vmin_vmax(results_to_plot, classes_left, classes_right, 'theta')
vmin_alpha, vmax_alpha = calculate_vmin_vmax(results_to_plot, classes_left, classes_right, 'alpha')
vmin_beta, vmax_beta = calculate_vmin_vmax(results_to_plot, classes_left, classes_right, 'beta')
vmin_gamma, vmax_gamma = calculate_vmin_vmax(results_to_plot, classes_left, classes_right, 'gamma')



def plot_4_hist(data, band, vmin, vmax):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.heatmap(np.array([[data[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['left_attleft'][0] for class_right in classes_right] for class_left in classes_left]), ax=ax[0, 0], annot=False, fmt=".2f", cmap='viridis', cbar_kws={'label': 'coherence'}, vmin = vmin, vmax = vmax)
    ax[0,0].set_xticks(np.arange(0, 20))
    ax[0,0].set_yticks(np.arange(0, 20))
    ax[0,0].set_xticklabels(classes_right)
    ax[0,0].set_yticklabels(classes_left)
    ax[0,0].set_xlabel('Right class')
    ax[0,0].set_ylabel('Left class')
    ax[0,0].set_title(f'coherence left w/ attending left - {band} band')

    sns.heatmap(np.array([[data[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['right_attleft'][0] for class_right in classes_right] for class_left in classes_left]), ax=ax[1,0], annot=False, fmt=".2f", cmap='viridis', cbar_kws={'label': 'coherence'}, vmin = vmin, vmax = vmax)
    ax[1,0].set_xticks(np.arange(0, 20))
    ax[1,0].set_yticks(np.arange(0, 20))
    ax[1,0].set_xticklabels(classes_right)
    ax[1,0].set_yticklabels(classes_left)
    ax[1,0].set_xlabel('Right class')
    ax[1,0].set_ylabel('Left class')
    ax[1,0].set_title(f'coherence right w/ attending left - {band} band')

    sns.heatmap(np.array([[data[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['left_attright'][0] for class_right in classes_right] for class_left in classes_left]), ax=ax[0,1], annot=False, fmt=".2f", cmap='viridis', cbar_kws={'label': 'coherence'}, vmin = vmin, vmax = vmax)
    ax[0,1].set_xticks(np.arange(0, 20))
    ax[0,1].set_yticks(np.arange(0, 20))
    ax[0,1].set_xticklabels(classes_right)
    ax[0,1].set_yticklabels(classes_left)
    ax[0,1].set_xlabel('Right class')
    ax[0,1].set_ylabel('Left class')
    ax[0,1].set_title(f'coherence left w/ attending right - {band} band')

    sns.heatmap(np.array([[data[f'left_class_{class_left}'][f'right_class_{class_right}'][band]['right_attright'][0] for class_right in classes_right] for class_left in classes_left]), ax=ax[1,1], annot=False, fmt=".2f", cmap='viridis', cbar_kws={'label': 'coherence'}, vmin = vmin, vmax = vmax)
    ax[1,1].set_xticks(np.arange(0, 20))
    ax[1,1].set_yticks(np.arange(0, 20))
    ax[1,1].set_xticklabels(classes_right)
    ax[1,1].set_yticklabels(classes_left)
    ax[1,1].set_xlabel('Right class')
    ax[1,1].set_ylabel('Left class')
    ax[1,1].set_title(f'coherence right w/ attending right - {band} band')

    plt.show()

plot_4_hist(results_to_plot, 'delta', vmin_delta, vmax_delta)
plot_4_hist(results_to_plot, 'theta',  vmin_theta, vmax_theta)
plot_4_hist(results_to_plot, 'alpha', vmin_alpha, vmax_alpha)
plot_4_hist(results_to_plot, 'beta', vmin_beta, vmax_beta)
plot_4_hist(results_to_plot, 'gamma', vmin_gamma, vmax_gamma)