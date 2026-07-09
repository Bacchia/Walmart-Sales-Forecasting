import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DLinear(nn.Module):

    def __init__(self, seq_len, pred_len):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.kernel_size = 5
        self.channels = 1
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x_in = x.squeeze(-1)
        
        trend_mean = x_in.mean(dim=1, keepdim=True)
        trend = trend_mean.repeat(1, self.seq_len)
        seasonal = x_in - trend
        
        pred_seasonal = self.linear_seasonal(seasonal)
        pred_trend = self.linear_trend(trend)
        
        final_output = pred_seasonal + pred_trend
        return final_output.unsqueeze(-1) 

def create_sequences_from_tabular(y_series, seq_len, pred_len):
    X_seq, y_seq = [], []
    for i in range(len(y_series) - seq_len - pred_len + 1):
        X_seq.append(y_series[i : i + seq_len])
        y_seq.append(y_series[i + seq_len : i + seq_len + pred_len])
    return (torch.tensor(np.array(X_seq), dtype=torch.float32).unsqueeze(-1), 
            torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(-1))

def train_dlinear_pipeline(y_train_scaled, params):
    X_train_t, y_train_t = create_sequences_from_tabular(
        y_train_scaled, params['seq_len'], params['pred_len']
    )
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    model = DLinear(params['seq_len'], params['pred_len'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(params['max_epochs']):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    return model
