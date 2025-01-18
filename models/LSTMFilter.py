import torch
import torch.nn as nn
from layers.lstm import LSTM

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = int(configs.window_length / 2)
        self.enc_dim = self.token_len
        self.ed_ratio = int(configs.state_dim / configs.obs_dim)
        self.dec_dim = self.ed_ratio * self.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        self.hidden_dim_of_lstm = 512
        self.hidden_layers_of_lstm = 4
        
        self.lstm_layer = LSTM(self.enc_dim, self.dec_dim, 
                             self.hidden_dim_of_lstm, self.hidden_layers_of_lstm)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, seq_len, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len] 
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)

        # Use LSTM layer for prediction
        predictions_flat = self.lstm_layer(x_enc)  # Pass through LSTM layer
        state_vars = n_vars * self.ed_ratio
        # Reshape predictions back to original batch size and variable dimensions
        predictions = predictions_flat.view(bs, state_vars, -1).permute(0, 2, 1)  # [bs, seq_len, n_vars]

        return predictions.to(torch.float32)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)