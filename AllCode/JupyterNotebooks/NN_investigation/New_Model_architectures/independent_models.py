import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Bidirectional LSTM with Attention
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=250, num_layers=2, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch, time, features]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch, time, hidden*2]
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Apply dropout and final linear layer
        context = self.dropout(context)
        output = self.fc(context)
        
        return output, attn_weights

# Temporal Convolutional Network (TCN)
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim=250, num_channels=[64, 128, 64, 32], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_dim)
        
    def forward(self, x):
        # x shape: [batch, time, features]
        # TCN expects [batch, features, time]
        x = x.transpose(1, 2)
        
        # Apply TCN
        y = self.tcn(x)
        
        # Global average pooling over time dimension
        y = y.mean(dim=2)
        
        # Final linear layer
        output = self.linear(y)
        
        return output, None  # Return None for attention weights to match interface

# CNN-LSTM Hybrid
class CNNLSTMHybrid(nn.Module):
    def __init__(self, input_dim, output_dim=250, cnn_channels=[32, 64], lstm_hidden=128, 
                 kernel_size=3, num_layers=2, dropout=0.3):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN layers for feature extraction
        cnn_layers = []
        in_channels = input_dim
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        
        # Output layer
        self.fc = nn.Linear(lstm_hidden * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch, time, features]
        # CNN expects [batch, features, time]
        x = x.transpose(1, 2)
        
        # Apply CNN for feature extraction
        cnn_out = self.cnn(x)
        
        # Reshape for LSTM: [batch, time, features]
        lstm_in = cnn_out.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Apply attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Apply dropout and final linear layer
        context = self.dropout(context)
        output = self.fc(context)
        
        return output, attn_weights

# Simple MLP
class SimpleAvgMLP(nn.Module):
    def __init__(self, input_dim, output_dim=250, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Create sequential model for hidden layers
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        # x: [batch, time, neurons]
        # collapse time by mean:
        pooled = x.mean(dim=1)         # [batch, neurons]
        h = self.hidden_layers(pooled)  # Apply all hidden layers
        out = self.output_layer(h)      # [batch, output_dim]
        return out, None  # Return None for attention weights to match LSTM interface
