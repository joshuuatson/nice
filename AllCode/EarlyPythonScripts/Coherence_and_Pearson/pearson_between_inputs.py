import numpy as np
import mne
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.signal import detrend
from numpy.polynomial.polynomial import Polynomial
import time
import gc
import seaborn as sns

np.set_printoptions(threshold=100)  # Default threshold
mne.set_log_level('WARNING') 
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

file_numbers = [1, 2, 4, 8, 14, 15, 20, 23]

classes_left = np.arange(0, 20)
classes_right = np.arange(0, 20)

def preprocess(data):
    # data = detrend(data, axis=0)
    # epsilon = 1e-9
    # data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + epsilon)
    
    return data

results_raw = {f'dataset_{file_number}': {f'left_class_{class_left}': {f'right_class_{class_right}': [] for class_right in classes_right} for class_left in classes_left} for file_number in file_numbers}
errors_raw = {f'dataset_{file_number}': {f'left_class_{class_left}': {f'right_class_{class_right}': [] for class_right in classes_right} for class_left in classes_left} for file_number in file_numbers}


def store_data(data_left, data_right, file_number, class_left, class_right):
    pearson_correlations = []

    for i in range(len(left_input_LFP_LR)):
        pearson_correlations.append(pearsonr(data_left[i], data_right[i])[0])

    mean_pearson = np.mean(pearson_correlations)  #takes the mean across trials and leaves a single value .float
    std = np.std(pearson_correlations)
    sem = std / np.sqrt(len(pearson_correlations))

    results_raw[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(mean_pearson)  #raw[dataset] gets a single values for each pair
    errors_raw[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'].append(sem)


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
    relevant = np.where(omitted == 0)[0]    #indices of agg where not omitted
    
    left_input_LFP = preprocess(data['LFP'][0][0][relevant])
    right_input_LFP = preprocess(data['LFP'][0][1][relevant])
    attention_LFP = preprocess(data['LFP_rec'][0][2][relevant])

    
    for class_left in classes_left:        
        for class_right in classes_right:

            
            indices_agg = np.where((omitted ==0) & (label_left == class_left) & (label_right == class_right))[0]
            indices = np.where(np.isin(relevant, indices_agg))[0]
                       
            if len(indices) == 0:
                continue
            else:
                #print(f' found data for left {class_left}, right {class_right}')
                left_input_LFP_LR = left_input_LFP[indices, :]
                right_input_LFP_LR = right_input_LFP[indices, :]

                store_data(left_input_LFP_LR, right_input_LFP_LR, file_number, class_left, class_right)
  

    del(data)
    gc.collect()
    print(f'Dataset {file_number} complete in', time.time() - file_total)

print('total time elapsed =', time.time() - total_time)

n_right = len(classes_right)  # number of right classes (e.g., 20)

results_to_plot = {f'left_class_{class_left}': {f'right_class_{class_right}': [] for class_right in classes_right} for class_left in classes_left}
errors_to_plot = {f'left_class_{class_left}': {f'right_class_{class_right}': [] for class_right in classes_right} for class_left in classes_left}

for class_left in classes_left:
    for class_right in classes_right:
        values = []
        errors = []
        for file_number in file_numbers:
            if results_raw[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}']:  #for given class pair, if there is a value for that dataset add it to the array
                values.append(results_raw[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'][0])  #value array should have 8 values
                errors.append(errors_raw[f'dataset_{file_number}'][f'left_class_{class_left}'][f'right_class_{class_right}'][0])

        if values:
            results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.mean(values)
            errors_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.mean(errors)
        else:
            results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.nan
            errors_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] = np.nan


max_value =  np.max(np.array([[results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] for class_right in classes_right] for class_left in classes_left]))
min_value = np.min(np.array([[results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] for class_right in classes_right] for class_left in classes_left]))

#---plotting the results------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(np.array([[results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'] for class_right in classes_right] for class_left in classes_left]), ax=ax, annot=False, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Pearson correlation'}, vmin=min_value, vmax=max_value)
ax.set_xticklabels(classes_right)
ax.set_yticklabels(classes_left)
ax.set_xlabel('Right class')
ax.set_ylabel('Left class')
ax.set_title('Pearson correlation between left and right inputs')
plt.show()

hist_values = []
for class_left in classes_left:
    for class_right in classes_right:
        hist_values.append(results_to_plot[f'left_class_{class_left}'][f'right_class_{class_right}'])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.histplot(hist_values, bins=20, kde=True, ax=ax)
ax.set_xlabel('Pearson correlation')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Pearson correlations between left and right inputs')
plt.show()

#pulling the classes for the lowest 50% of values:
lowest_values = np.array(hist_values).argsort()[:int(0.2 * len(hist_values))]
#lowest_values = np.array(hist_values).argsort()[:]
lowest_values = [(classes_left[i // n_right], classes_right[i % n_right]) for i in lowest_values]   #// is floor division, % is modulo
print('Lowest 20% of values:', lowest_values)

#Lowest 50% of values: [(np.int64(12), np.int64(18)), (np.int64(18), np.int64(12)), (np.int64(7), np.int64(18)), (np.int64(14), np.int64(18)), (np.int64(18), np.int64(7)), (np.int64(12), np.int64(8)), (np.int64(8), np.int64(12)), (np.int64(7), np.int64(8)), (np.int64(8), np.int64(7)), (np.int64(18), np.int64(15)), (np.int64(16), np.int64(12)), (np.int64(18), np.int64(14)), (np.int64(6), np.int64(18)), (np.int64(15), np.int64(18)), (np.int64(8), np.int64(14)), (np.int64(11), np.int64(7)), (np.int64(17), np.int64(18)), (np.int64(7), np.int64(11)), (np.int64(16), np.int64(7)), (np.int64(3), np.int64(12)), (np.int64(11), np.int64(12)), (np.int64(18), np.int64(17)), (np.int64(18), np.int64(6)), (np.int64(12), np.int64(10)), (np.int64(12), np.int64(11)), (np.int64(0), np.int64(18)), (np.int64(18), np.int64(2)), (np.int64(7), np.int64(10)), (np.int64(7), np.int64(16)), (np.int64(2), np.int64(18)), (np.int64(18), np.int64(0)), (np.int64(12), np.int64(6)), (np.int64(7), np.int64(1)), (np.int64(6), np.int64(8)), (np.int64(12), np.int64(16)), (np.int64(12), np.int64(1)), (np.int64(7), np.int64(19)), (np.int64(17), np.int64(8)), (np.int64(10), np.int64(7)), (np.int64(18), np.int64(13)), (np.int64(6), np.int64(12)), (np.int64(12), np.int64(19)), (np.int64(19), np.int64(12)), (np.int64(12), np.int64(5)), (np.int64(9), np.int64(12)), (np.int64(4), np.int64(18)), (np.int64(10), np.int64(12)), (np.int64(18), np.int64(4)), (np.int64(6), np.int64(17)), (np.int64(8), np.int64(17)), (np.int64(16), np.int64(17)), (np.int64(16), np.int64(15)), (np.int64(8), np.int64(6)), (np.int64(7), np.int64(9)), (np.int64(7), np.int64(5)), (np.int64(17), np.int64(6)), (np.int64(17), np.int64(16)), (np.int64(11), np.int64(14)), (np.int64(18), np.int64(5)), (np.int64(7), np.int64(6)), (np.int64(6), np.int64(10)), (np.int64(13), np.int64(18)), (np.int64(14), np.int64(8)), (np.int64(18), np.int64(3)), (np.int64(8), np.int64(0)), (np.int64(6), np.int64(5)), (np.int64(1), np.int64(7)), (np.int64(15), np.int64(8)), (np.int64(16), np.int64(14)), (np.int64(0), np.int64(8)), (np.int64(8), np.int64(2)), (np.int64(1), np.int64(12)), (np.int64(12), np.int64(3)), (np.int64(16), np.int64(0)), (np.int64(19), np.int64(7)), (np.int64(19), np.int64(6)), (np.int64(5), np.int64(7)), (np.int64(3), np.int64(7)), (np.int64(19), np.int64(17)), (np.int64(9), np.int64(7))]

#highest 12- 18
#lowest 19-9 but also several diagonals, e.g. 88, 11 11, 19 19 ...