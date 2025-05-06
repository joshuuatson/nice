import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
import pickle
import torch.nn as nn
from scipy.stats import zscore
from nice.algorithms.connectivity_AT import *
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from torch.utils.data import DataLoader, TensorDataset


file_numbers = [1]
#file_numbers = [1, 2, 4, 8, 14, 15, 20, 23] #for the full dataset
structure = {f'dataset_{file_number}': {
    'left_attleft': [],
    'right_attleft': [],
    'left_attright': [],
    'right_attright': []
} for file_number in file_numbers}


def preprocess(data):
    #data shape: (trials, time, neurons)
    trials, time, num_neurons = data.shape
    z = np.zeros_like(data)

    for neuron in range(num_neurons):
        for t in range(time):
            vals = data[:, t, neuron]  #this is all the data for given neuron at given time point
            std = np.std(vals)
            if std > 1e-6:
                z[:, t, neuron] = (vals - np.mean(vals)) / std    #zscore if there is a stdev
            else:
                z[:, t, neuron] = np.median(vals)   #returnt the median value otherwise  (median rather than mean in case of large variability in a singel point)

    if np.isnan(z).any():
        raise ValueError("Data contains NaN values after normalization.")

    return z


def get_data(file_number):
    file_path = f'C:/Users/joshu/PartIIIProject/RSNNdale_attention_{file_number}_attention_test'
    data = pickle.load(open(file_path, 'rb'))
  
    label_left = data['label_left'][0]
    label_right = data['label_right'][0]
    attend_01 = data['attend'][0]
    omitted = data['omit'][0]
 
    left_indices_agg = np.where((omitted ==0) & (attend_01 == 0) & (label_left != label_right))[0]  #indices of agg where attention left
    right_indices_agg = np.where((omitted ==0) & (attend_01 == 1) & (label_left != label_right))[0]  #indices of agg where attention right

    left_attleft = data['SP'][0][0][left_indices_agg, 100:350, :]
    print('lal shape:', left_attleft.shape)
    left_attright = data['SP'][0][0][right_indices_agg, 100:350, :]
    print('lar shape:', left_attright.shape)

    right_attleft = data['SP'][0][1][left_indices_agg, 100:350, :]
    right_attright = data['SP'][0][1][right_indices_agg, 100:350, :]

    att_attleft = data['SP'][0][2][left_indices_agg, 100:350, :]
    att_attright = data['SP'][0][2][right_indices_agg, 100:350, :]


    num_trials_left, num_samples, num_neurons = left_attleft.shape
    num_trials_right = left_attright.shape[0]
    num_neurons_attention = 80

    return left_attleft, left_attright, right_attleft, right_attright, att_attleft, att_attright, num_trials_left, num_trials_right, num_samples, num_neurons, num_neurons_attention
##e.g. dataset 1 gives left_attleft of shape (469 trials, 500 time points, 160 spikes)

l_al = []
l_ar = []
r_al = []
r_ar = []
a_al = []
a_ar = []

def collect_data():
    for file_number in file_numbers:
        left_attleft, left_attright, right_attleft, right_attright, att_attleft, att_attright, _, _, _, _, _ = get_data(file_number)
        l_al.append(left_attleft)
        l_ar.append(left_attright)
        r_al.append(right_attleft)
        r_ar.append(right_attright)
        a_al.append(att_attleft)
        a_ar.append(att_attright)
       
    return l_al, l_ar, r_al, r_ar, a_al, a_ar
   

l_al, _, _, _, a_al, _ = collect_data()

l_al_all = np.concatenate(l_al, axis=0)
a_al_all = np.concatenate(a_al, axis=0)
print('l_al_all shape:', l_al_all.shape)
print('a_al_all shape:', a_al_all.shape)

x = torch.tensor(l_al_all, dtype=torch.float32)  #l_al shape (n_trials, n_time, n_neurons)
y = torch.tensor(a_al_all, dtype=torch.float32)  #a_al shape (n_trials, n_time, n_neurons)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

train_datast = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_datast, batch_size=8, shuffle=True, drop_last = True, num_workers=0)


class SpikingAttentionLSTM(nn.Module):
    def __init__(self, input_dim = 160, hidden_dim = 64, output_dim = 80, num_layers = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim, 
            batch_first = True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.lstm(x) #shape (batch_size, 500, hidden_dim)
        out = self.fc(output)
        return out   #this will be shape (batch_size, 500, 80)

model1 = SpikingAttentionLSTM(input_dim = 160, hidden_dim = 64, output_dim = 80, num_layers = 1)

#training loop
loss_history = []
criterion = nn.MSELoss()
def train(model, train_loader, criterion, epochs = 1000):
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    for _ in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()


        loss_history.append(epoch_loss / len(train_loader))
        if _ % 100 == 0: 
            print(f'Epoch {_}, Loss: {epoch_loss / len(train_loader)}')

train(model1, train_loader, criterion, epochs = 1000)
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

y_pred_trained = model1(x_test).detach().numpy()
print('y_pred shape:', y_pred_trained.shape)

