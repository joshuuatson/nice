import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from nice.algorithms.connectivity import epochs_compute_wsmi
np.set_printoptions(threshold=100)  # Default threshold
mne.set_log_level('WARNING') 
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.signal import detrend
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import ttest_rel
import time
import gc
import seaborn as sns

file_numbers = [1, 2, 4, 8, 14, 15, 20, 23]

classes_left = np.arange(0, 20)
classes_right = np.arange(0, 20)

results_raw = {}
errors_raw = {}

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
    omitted = data["omit"][0]

    # Extract LFP signals
    left_input_LFP = data['LFP'][0][0]
    right_input_LFP = data['LFP'][0][1]
    attention_LFP = data['LFP_rec'][0][2]

    #want to load a dataset, and then do the calculations - can then store pearson for 4 conditions, 20 combinations for each dataset
    #will take mean over datasets, leaving (4, 20)
    #for each dataset, will need to loop over left class and right class
    results_raw[f'dataset_{file_number}'] = {
        'left_attleft': {},
        'right_attleft': {}, 
        'left_attright': {},
        'right_attright': {}
    }

    errors_raw[f'dataset_{file_number}'] = {
        'left_attleft': {},
        'right_attleft': {}, 
        'left_attright': {},
        'right_attright': {}
    }

    for class_left in classes_left:
        results_raw[f'dataset_{file_number}']['left_attleft'][f'left_class_{class_left}'] = {}
        results_raw[f'dataset_{file_number}']['right_attleft'][f'left_class_{class_left}'] = {}
        results_raw[f'dataset_{file_number}']['left_attright'][f'left_class_{class_left}'] = {}
        results_raw[f'dataset_{file_number}']['right_attright'][f'left_class_{class_left}'] = {}

        errors_raw[f'dataset_{file_number}']['left_attleft'][f'left_class_{class_left}'] = {}
        errors_raw[f'dataset_{file_number}']['right_attleft'][f'left_class_{class_left}'] = {}
        errors_raw[f'dataset_{file_number}']['left_attright'][f'left_class_{class_left}'] = {}
        errors_raw[f'dataset_{file_number}']['right_attright'][f'left_class_{class_left}'] = {}

        
        for class_right in classes_right:
            results_raw[f'dataset_{file_number}']['left_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
            results_raw[f'dataset_{file_number}']['right_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
            results_raw[f'dataset_{file_number}']['left_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
            results_raw[f'dataset_{file_number}']['right_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []

            errors_raw[f'dataset_{file_number}']['left_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
            errors_raw[f'dataset_{file_number}']['right_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
            errors_raw[f'dataset_{file_number}']['left_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
            errors_raw[f'dataset_{file_number}']['right_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'] = []
         
         

            left_indices = np.where((omitted ==0) & (attend_01 == 0) & (label_left == class_left) & (label_right == class_right))[0]
            right_indices = np.where((omitted ==0) & (attend_01 == 1) & (label_left == class_left) & (label_right == class_right))[0]

            
                       
            if len(left_indices) == 0 or len(right_indices) == 0:
                #
                continue
            else:
                #print(f' found data for left {class_left}, right {class_right}')
                left_input_LFP_attleft = left_input_LFP[left_indices]
                right_input_LFP_attleft = right_input_LFP[left_indices]
                attention_LFP_attleft = attention_LFP[left_indices]

                left_input_LFP_attright = left_input_LFP[right_indices]
                right_input_LFP_attright = right_input_LFP[right_indices]
                attention_LFP_attright = attention_LFP[right_indices]

             
                #--- preprocessing the data ----------------------------------------NB detrending first to avoid 0 and NaN values
                if len(left_indices) >= 1 and len(right_indices) >= 1:
                    left_input_LFP_attleft = detrend(left_input_LFP_attleft, axis=0)
                    right_input_LFP_attleft = detrend(right_input_LFP_attleft, axis=0)
                    attention_LFP_attleft = detrend(attention_LFP_attleft, axis=0)

                    left_input_LFP_attright = detrend(left_input_LFP_attright, axis=0)
                    right_input_LFP_attright = detrend(right_input_LFP_attright, axis=0)
                    attention_LFP_attright = detrend(attention_LFP_attright, axis=0)

                for i in range(len(left_input_LFP_attleft)):
                    left_input_LFP_attleft[i] = detrend(left_input_LFP_attleft[i])
                    right_input_LFP_attleft[i] = detrend(right_input_LFP_attleft[i])
                    attention_LFP_attleft[i] = detrend(attention_LFP_attleft[i])

                for i in range(len(left_input_LFP_attright)):
                    left_input_LFP_attright[i] = detrend(left_input_LFP_attright[i])
                    right_input_LFP_attright[i] = detrend(right_input_LFP_attright[i])
                    attention_LFP_attright[i] = detrend(attention_LFP_attright[i])

                if len(left_indices) >= 2 and len(right_indices) >= 2:
                    left_input_LFP_attleft = zscore(left_input_LFP_attleft, axis=0)
                    right_input_LFP_attleft = zscore(right_input_LFP_attleft, axis=0)
                    attention_LFP_attleft = zscore(attention_LFP_attleft, axis=0)
                
                    left_input_LFP_attright = zscore(left_input_LFP_attright, axis=0)
                    right_input_LFP_attright = zscore(right_input_LFP_attright, axis=0)
                    attention_LFP_attright = zscore(attention_LFP_attright, axis=0)

                #----------------------------------------------------------------
                left_attleft_pearson = []
                right_attleft_pearson = []
                left_attright_pearson = []
                right_attright_pearson = []

                for i in range(len(left_input_LFP_attleft)):
                    left_attleft_pearson.append(pearsonr(left_input_LFP_attleft[i], attention_LFP_attleft[i])[0])
                    right_attleft_pearson.append(pearsonr(right_input_LFP_attleft[i], attention_LFP_attleft[i])[0])

                for i in range(len(left_input_LFP_attright)):
                    left_attright_pearson.append(pearsonr(left_input_LFP_attright[i], attention_LFP_attright[i])[0])
                    right_attright_pearson.append(pearsonr(right_input_LFP_attright[i], attention_LFP_attright[i])[0])

                mean_left_attleft = np.mean(left_attleft_pearson)
                mean_right_attleft = np.mean(right_attleft_pearson)
                mean_left_attright = np.mean(left_attright_pearson)
                mean_right_attright = np.mean(right_attright_pearson)

                std_left_attleft = np.std(left_attleft_pearson)
                std_right_attleft = np.std(right_attleft_pearson)
                std_left_attright = np.std(left_attright_pearson)
                std_right_attright = np.std(right_attright_pearson)

                sem_left_attleft = std_left_attleft / np.sqrt(len(left_attleft_pearson))
                sem_right_attleft = std_right_attleft / np.sqrt(len(right_attleft_pearson))
                sem_left_attright = std_left_attright / np.sqrt(len(left_attright_pearson))
                sem_right_attright = std_right_attright / np.sqrt(len(right_attright_pearson))

                #-------storing these means across trials into results dictionary-------------
                #have given resuylts left a dict for this right class, so append to that class

                results_raw[f'dataset_{file_number}']['left_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(mean_left_attleft)
                results_raw[f'dataset_{file_number}']['right_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(mean_right_attleft)
                results_raw[f'dataset_{file_number}']['left_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(mean_left_attright)
                results_raw[f'dataset_{file_number}']['right_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(mean_right_attright)

                errors_raw[f'dataset_{file_number}']['left_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(sem_left_attleft)
                errors_raw[f'dataset_{file_number}']['right_attleft'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(sem_right_attleft)
                errors_raw[f'dataset_{file_number}']['left_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(sem_left_attright)
                errors_raw[f'dataset_{file_number}']['right_attright'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(sem_right_attright)

                del left_input_LFP_attleft, right_input_LFP_attleft, attention_LFP_attleft, left_input_LFP_attright, right_input_LFP_attright, attention_LFP_attright
                gc.collect()
    del(data)
    gc.collect()
    print(f'Dataset {file_number} complete in', time.time() - file_total)

print('total time elapsed =', time.time() - total_time)


results_to_plot = {}  # dict with each left_class: dict of conditions -> averaged pearson (array of length=len(classes_right))
errors_to_plot = {}   # same structure for errors

n_right = len(classes_right)  # number of right classes (e.g., 20)

for class_left in classes_left:
    results_to_plot[f'class_left_{class_left}'] = {
        'left_attleft': [],
        'right_attleft': [],
        'left_attright': [],
        'right_attright': []
    }
    errors_to_plot[f'class_left_{class_left}'] = {
        'left_attleft': [],
        'right_attleft': [],
        'left_attright': [],
        'right_attright': []
    }

#want to loop over each left class, and then for each right class, take the mean across datasets
#and append to the results_to_plot dict
for class_left in classes_left:
    for condition in ['left_attleft', 'right_attleft', 'left_attright', 'right_attright']:
        for class_right in classes_right:
            values = []
            errors = []
            for file_number in file_numbers:
                if results_raw[f'dataset_{file_number}'][condition][f'left_class_{class_left}'][f'right_class_{class_right}']:
                    values.append(results_raw[f'dataset_{file_number}'][condition][f'left_class_{class_left}'][f'right_class_{class_right}'][0])
                    errors.append(errors_raw[f'dataset_{file_number}'][condition][f'left_class_{class_left}'][f'right_class_{class_right}'][0])
            if values:
                results_to_plot[f'class_left_{class_left}'][condition].append(np.mean(values))
                errors_to_plot[f'class_left_{class_left}'][condition].append(np.mean(errors))
            else:
                results_to_plot[f'class_left_{class_left}'][condition].append(np.nan)
                errors_to_plot[f'class_left_{class_left}'][condition].append(np.nan)

differences = {}

for class_left in classes_left:
    differences[f'class_left_{class_left}'] = {
        'attleft_diff': [],
        'attright_diff': []
    }

    for class_right in classes_right:
        left_attleft_value = results_to_plot[f'class_left_{class_left}']['left_attleft'][class_right]
        right_attleft_value = results_to_plot[f'class_left_{class_left}']['right_attleft'][class_right]
        left_attright_value = results_to_plot[f'class_left_{class_left}']['left_attright'][class_right]
        right_attright_value = results_to_plot[f'class_left_{class_left}']['right_attright'][class_right]

        attleft_diff = left_attleft_value - right_attleft_value
        attright_diff = left_attright_value - right_attright_value

        differences[f'class_left_{class_left}']['attleft_diff'].append(abs(attleft_diff))
        differences[f'class_left_{class_left}']['attright_diff'].append(abs(attright_diff))

# Plot the differences
for class_left in classes_left:
    figs, axs = plt.subplots(1, 2, figsize=(10, 5))

    attleft_diff_data = differences[f'class_left_{class_left}']['attleft_diff']
    attright_diff_data = differences[f'class_left_{class_left}']['attright_diff']

    axs[0].plot(classes_right, attleft_diff_data, '-x', color='b', label='attleft_diff')
    axs[0].set_title(f'Attention Left Difference - left class = {class_left}')
    axs[0].set_xlabel('Class Right')
    axs[0].set_ylabel('Difference in Pearson Correlation')
    axs[0].legend()

    axs[1].plot(classes_right, attright_diff_data, '-x', color='b', label='attright_diff')
    axs[1].set_title(f'Attention Right Difference - left class = {class_left}')
    axs[1].set_xlabel('Class Right')
    axs[1].set_ylabel('Difference in Pearson Correlation')
    axs[1].legend()

    plt.tight_layout()
    # Save the figures
    figs.savefig(f'C:/Users/joshu/OneDrive/Documents/Physics/PartIII/Project/180225_different_class_plots/pearson_correlations_all_leftright/pearson_difference_plots_LRA/pearson_correlation_differences_{class_left}_diff.png')


# Create heatmaps for the differences

# Prepare data for heatmaps
attleft_diff_matrix = np.zeros((len(classes_left), len(classes_right)))
attright_diff_matrix = np.zeros((len(classes_left), len(classes_right)))

for i, class_left in enumerate(classes_left):
    attleft_diff_matrix[i, :] = differences[f'class_left_{class_left}']['attleft_diff']
    attright_diff_matrix[i, :] = differences[f'class_left_{class_left}']['attright_diff']

# Plot heatmaps
fig_heat, axs = plt.subplots(1, 2, figsize=(20, 10))

sns.heatmap(attleft_diff_matrix, ax=axs[0], cmap='viridis', xticklabels=classes_right, yticklabels=classes_left)
axs[0].set_title('Attention Left Difference Heatmap')
axs[0].set_xlabel('Class Right')
axs[0].set_ylabel('Class Left')

sns.heatmap(attright_diff_matrix, ax=axs[1], cmap='viridis', xticklabels=classes_right, yticklabels=classes_left)
axs[1].set_title('Attention Right Difference Heatmap')
axs[1].set_xlabel('Class Right')
axs[1].set_ylabel('Class Left')

plt.tight_layout()
# Save the figures
fig_heat.savefig('C:/Users/joshu/OneDrive/Documents/Physics/PartIII/Project/180225_different_class_plots/pearson_correlations_all_leftright/Heatmaps_LRA/Heatmap_for_pearson_inputsvsattention.png')

# for class_left in classes_left:
#     figs, axs = plt.subplots(1, 2, figsize=(10, 5))

#     # Retrieve data for this left class
#     left_attleft_data = results_to_plot[f'class_left_{class_left}']['left_attleft']
#     right_attleft_data = results_to_plot[f'class_left_{class_left}']['right_attleft']
#     left_attright_data = results_to_plot[f'class_left_{class_left}']['left_attright']
#     right_attright_data = results_to_plot[f'class_left_{class_left}']['right_attright']

#     error_left_attleft = errors_to_plot[f'class_left_{class_left}']['left_attleft']
#     error_right_attleft = errors_to_plot[f'class_left_{class_left}']['right_attleft']
#     error_left_attright = errors_to_plot[f'class_left_{class_left}']['left_attright']
#     error_right_attright = errors_to_plot[f'class_left_{class_left}']['right_attright']

#     axs[0].errorbar(classes_right, left_attleft_data, yerr=error_left_attleft,
#                     label='left_attleft', fmt='-x', color='r')
#     axs[0].errorbar(classes_right, right_attleft_data, yerr=error_right_attleft,
#                     label='right_attleft', fmt='-x', color='k')
#     axs[0].set_title(f'Attention Left - left class = {class_left}')
#     axs[0].set_xlabel('Class Right')
#     axs[0].set_ylabel('Mean Pearson Correlation')
#     axs[0].legend()

#     axs[1].errorbar(classes_right, left_attright_data, yerr=error_left_attright,
#                     label='left_attright', fmt='-x', color='r')
#     axs[1].errorbar(classes_right, right_attright_data, yerr=error_right_attright,
#                     label='right_attright', fmt='-x', color='k')
#     axs[1].set_title(f'Attention Right - left class = {class_left}')
#     axs[1].set_xlabel('Class Right')
#     axs[1].set_ylabel('Mean Pearson Correlation')
#     axs[1].legend()

#     plt.tight_layout()
#     #plt.show()
#     plt.tight_layout()

#     # Save the figures
#     output_folder = 'C:/Users/joshu/OneDrive/Documents/Physics/PartIII/Project/180225_different_class_plots/pearson_correlations_all_leftright/pearo'
#     figs.savefig(f'{output_folder}pearson_correlation_differences_{class_left}.png')


