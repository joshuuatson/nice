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

##---this calculates wsmi and pearson for LFP data
file_numbers = [1]
file_numbers = [1, 2, 4, 8, 14, 15, 20, 23]   #have dropped data 3
##now have data for the significance of wsmi values at each tau, between classes 0 and i (left, right)
##want to see if the pearson correlations between those classes are predictive of the significance

#for data set 3, error with left, but larger wsmi - seems indicative of a problem with the calculation
# file_numbers = [1]
# Initialize results dictionary for 10 datasets

#goal is to be able to specify the class combination and look at the pearson correlation across datasets
#want to then be able to just loop over different class combinations


classes_right = np.arange(0, 3)
results_classes = {}
errors_classes = {}


for class_right in classes_right:
    results_classes[f'class_{class_right}'] = {
        'left_attleft': [],
        'right_attleft': [], 
        'left_attright': [],
        'right_attright': []
    }

    errors_classes[f'class_{class_right}'] = {
        'left_attleft': [],
        'right_attleft': [], 
        'left_attright': [],
        'right_attright': []
    }
    
    results_dataset = {}
    errors_dataset = {}
    n_values = {}

    start_time_class = time.time()

    for file_number in file_numbers:
        results_dataset[f'dataset_{file_number}'] = {
            'left_attleft': [],
            'right_attleft': [], 
            'left_attright': [],
            'right_attright': []
        }
        
        errors_dataset[f'dataset_{file_number}'] = {
            'left_attleft': [],
            'right_attleft': [], 
            'left_attright': [],
            'right_attright': []
        }
  
        
        n_values[f'dataset_{file_number}'] = {
            'attleft': [], 
            'attright': []
        }

#--loading a dataset---
        start_time = time.time()
        file_path = f'C:/Users/joshu/PartIIIProject/RSNNdale_attention_{file_number}_attention_test'
        data = pickle.load(open(file_path, 'rb'))

        attention_labels = data['label_attend'][0]
        label_left = data['label_left'][0]
        label_right = data['label_right'][0]
        attend_01= data['attend'][0]

        #not filtered for omitted trials 
        left_input_LFP = data['LFP'][0][0]  # Left input  [0,1] means left, Right 
        right_input_LFP = data['LFP'][0][1]  # Right input
        attention_LFP = data['LFP_rec'][0][2]  # Attention layer  [2] means attention 
        omitted = data["omit"][0]

        elapsed_time = time.time() - start_time
        print(f"Dataset {file_number} loaded in {elapsed_time:.2f} seconds")


        left_indices = {}
        right_indices = {}

        for i in range(0, 20):
            left_indices[i] = np.where((omitted == 0) & (attend_01 == 0) & (label_left == 0) & (label_right == i))[0]
            right_indices[i] = np.where((omitted == 0) & (attend_01 == 1) & (label_left == 0) & (label_right == i))[0]

        if len(left_indices[class_right]) == 0 or len(right_indices[class_right]) == 0:
            continue

        #for left 0 and right ith
        left_input_LFP_om_left = left_input_LFP[left_indices[class_right]]
        left_input_LFP_om_left_relevant = left_input_LFP_om_left[:, 100:350]
        right_input_LFP_om_left = right_input_LFP[left_indices[class_right]]
        right_input_LFP_om_left_relevant = right_input_LFP_om_left[:, 100:350]
        attention_LFP_om_left = attention_LFP[left_indices[class_right]]
        attention_LFP_om_left_relevant = attention_LFP_om_left[:, 100:350]


        left_input_LFP_om_right = left_input_LFP[right_indices[class_right]]
        left_input_LFP_om_right_relevant = left_input_LFP_om_right[:, 100:350]
        right_input_LFP_om_right = right_input_LFP[right_indices[class_right]]
        right_input_LFP_om_right_relevant = right_input_LFP_om_right[:, 100:350]
        attention_LFP_om_right = attention_LFP[right_indices[class_right]]
        attention_LFP_om_right_relevant = attention_LFP_om_right[:, 100:350]

        
        n_values[f'dataset_{file_number}']['attleft'].append(len(left_input_LFP_om_left_relevant))
        n_values[f'dataset_{file_number}']['attright'].append(len(left_input_LFP_om_right_relevant))

        
#---preprocessing the data, zscore across trials and detrending across trials and time---
        left_input_LFP_om_left_relevant = detrend(left_input_LFP_om_left_relevant, axis=0)
        right_input_LFP_om_left_relevant = detrend(right_input_LFP_om_left_relevant, axis=0)
        attention_LFP_om_left_relevant = detrend(attention_LFP_om_left_relevant, axis=0)

        left_input_LFP_om_right_relevant = detrend(left_input_LFP_om_right_relevant, axis=0)
        right_input_LFP_om_right_relevant = detrend(right_input_LFP_om_right_relevant, axis=0)
        attention_LFP_om_right_relevant = detrend(attention_LFP_om_right_relevant, axis=0)


        #---detrend in time---
        for i in range(len(left_input_LFP_om_left_relevant)):
            left_input_LFP_om_left_relevant[i] = detrend(left_input_LFP_om_left_relevant[i])
            right_input_LFP_om_left_relevant[i] = detrend(right_input_LFP_om_left_relevant[i])
            attention_LFP_om_left_relevant[i] = detrend(attention_LFP_om_left_relevant[i])


        for i in range(len(left_input_LFP_om_right_relevant)):
            left_input_LFP_om_right_relevant[i] = detrend(left_input_LFP_om_right_relevant[i])
            right_input_LFP_om_right_relevant[i] = detrend(right_input_LFP_om_right_relevant[i])
            attention_LFP_om_right_relevant[i] = detrend(attention_LFP_om_right_relevant[i])


        ##---zscore across trials---
        left_input_LFP_om_left_relevant = zscore(left_input_LFP_om_left_relevant, axis=0)
        right_input_LFP_om_left_relevant = zscore(right_input_LFP_om_left_relevant, axis=0)
        attention_LFP_om_left_relevant = zscore(attention_LFP_om_left_relevant, axis=0)

        left_input_LFP_om_right_relevant = zscore(left_input_LFP_om_right_relevant, axis=0)
        right_input_LFP_om_right_relevant = zscore(right_input_LFP_om_right_relevant, axis=0)
        attention_LFP_om_right_relevant = zscore(attention_LFP_om_right_relevant, axis=0)


#--- looking at the pearson correlations---
        attleft_pearson_left = []
        attleft_pearson_right = []
        attright_pearson_left = []
        attright_pearson_right = []

        for i in range(len(left_input_LFP_om_left_relevant)):
            corr_left, _ = pearsonr(left_input_LFP_om_left_relevant[i], attention_LFP_om_left_relevant[i])
            attleft_pearson_left.append(corr_left)
            corr_right, _ = pearsonr(right_input_LFP_om_left_relevant[i], attention_LFP_om_left_relevant[i])
            attleft_pearson_right.append(corr_right)

        for i in range(len(left_input_LFP_om_right_relevant)):
            corr_left, _ = pearsonr(left_input_LFP_om_right_relevant[i], attention_LFP_om_right_relevant[i])
            attright_pearson_left.append(corr_left)
            corr_right, _ = pearsonr(right_input_LFP_om_right_relevant[i], attention_LFP_om_right_relevant[i])
            attright_pearson_right.append(corr_right)

#this gives the pearson correlations for each trial. will take the mean correlation across trials and that will give us the class-wise correlations     
        mean_corr_left_attleft = np.mean(attleft_pearson_left)
        mean_corr_right_attleft = np.mean(attleft_pearson_right)
        mean_corr_left_attright = np.mean(attright_pearson_left)
        mean_corr_right_attright = np.mean(attright_pearson_right)

        stdev_corr_left_attleft = np.std(attleft_pearson_left)
        stdev_corr_right_attleft = np.std(attleft_pearson_right)
        stdev_corr_left_attright = np.std(attright_pearson_left)
        stdev_corr_right_attright = np.std(attright_pearson_right)

        SEM_left_attleft = stdev_corr_left_attleft / np.sqrt(n_values[f'dataset_{file_number}']['attleft'][0])
        SEM_right_attleft = stdev_corr_right_attleft / np.sqrt(n_values[f'dataset_{file_number}']['attleft'][0])
        SEM_left_attright = stdev_corr_left_attright / np.sqrt(n_values[f'dataset_{file_number}']['attright'][0])
        SEM_right_attright = stdev_corr_right_attright / np.sqrt(n_values[f'dataset_{file_number}']['attright'][0])


#--anticipating that for lower correlations, the wsmi values will be less significant (for the later classes)
#want to store these for each dataset, so that mean across datasets can be taken for given class combination

        results_dataset[f'dataset_{file_number}']['left_attleft'].append(mean_corr_left_attleft)
        results_dataset[f'dataset_{file_number}']['right_attleft'].append(mean_corr_right_attleft)
        results_dataset[f'dataset_{file_number}']['left_attright'].append(mean_corr_left_attright)
        results_dataset[f'dataset_{file_number}']['right_attright'].append(mean_corr_right_attright)


        errors_dataset[f'dataset_{file_number}']['left_attleft'].append(SEM_left_attleft)
        errors_dataset[f'dataset_{file_number}']['right_attleft'].append(SEM_right_attleft)
        errors_dataset[f'dataset_{file_number}']['left_attright'].append(SEM_left_attright)
        errors_dataset[f'dataset_{file_number}']['right_attright'].append(SEM_right_attright)

        del data, left_input_LFP, right_input_LFP, attention_LFP
        del left_input_LFP_om_left, right_input_LFP_om_left, attention_LFP_om_left
        del left_input_LFP_om_right, right_input_LFP_om_right, attention_LFP_om_right
        gc.collect()  
   

#--- now have values and errors for each dataset, want to take mean across datasets to plot for each class combination
    results_classes[f'class_{class_right}']['left_attleft'].append(np.mean(results_dataset[f'dataset_{file_number}']['left_attleft']))
    results_classes[f'class_{class_right}']['right_attleft'].append(np.mean(results_dataset[f'dataset_{file_number}']['right_attleft']))
    results_classes[f'class_{class_right}']['left_attright'].append(np.mean(results_dataset[f'dataset_{file_number}']['left_attright']))
    results_classes[f'class_{class_right}']['right_attright'].append(np.mean(results_dataset[f'dataset_{file_number}']['right_attright']))

    errors_classes[f'class_{class_right}']['left_attleft'].append(np.mean(errors_dataset[f'dataset_{file_number}']['left_attleft']))
    errors_classes[f'class_{class_right}']['right_attleft'].append(np.mean(errors_dataset[f'dataset_{file_number}']['right_attleft']))
    errors_classes[f'class_{class_right}']['left_attright'].append(np.mean(errors_dataset[f'dataset_{file_number}']['left_attright']))
    errors_classes[f'class_{class_right}']['right_attright'].append(np.mean(errors_dataset[f'dataset_{file_number}']['right_attright']))


    elapsed_time = time.time() - start_time_class
    print(f"Class 0{class_right} finished in {elapsed_time:.2f} seconds")
    

       




left_attleft_data = []
right_attleft_data = []
left_attright_data = []
right_attright_data = []

for i in range(len(results_classes)):
    left_attleft_data.append(results_classes[f'class_{i}']['left_attleft'][0])
    right_attleft_data.append(results_classes[f'class_{i}']['right_attleft'][0])
    left_attright_data.append(results_classes[f'class_{i}']['left_attright'][0])
    right_attright_data.append(results_classes[f'class_{i}']['right_attright'][0])

error_left_attleft = []
error_right_attleft = []
error_left_attright = []
error_right_attright = []

for i in range(len(errors_classes)):
    error_left_attleft.append(errors_classes[f'class_{i}']['left_attleft'][0])
    error_right_attleft.append(errors_classes[f'class_{i}']['right_attleft'][0])
    error_left_attright.append(errors_classes[f'class_{i}']['left_attright'][0])
    error_right_attright.append(errors_classes[f'class_{i}']['right_attright'][0])


#---plotting the results---
figs, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot for Attention Left
axs[0].errorbar(classes_right, left_attleft_data, yerr=error_left_attleft, label='left_attleft', fmt='-o', color = 'r')
axs[0].errorbar(classes_right, right_attleft_data, yerr=error_right_attleft, label='right_attleft', fmt='-o', color = 'k')
axs[0].set_title('Attention Left - left class = 0')
axs[0].set_xlabel('Class Right')
axs[0].set_ylabel('Mean Pearson Correlation')
axs[0].legend()

# Plot for Attention Right
axs[1].errorbar(classes_right, left_attright_data, yerr=error_left_attright, label='left_attright', fmt='-o', color = 'r')
axs[1].errorbar(classes_right, right_attright_data, yerr=error_right_attright, label='right_attright', fmt='-o', color = 'k')
axs[1].set_title('Attention Right - left class = 0')
axs[1].set_xlabel('Class Right')
axs[1].set_ylabel('Mean Pearson Correlation')
axs[1].legend()

plt.tight_layout()
plt.show()                