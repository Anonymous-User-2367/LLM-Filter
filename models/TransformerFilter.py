import torch
import torch.nn as nn
from layers.mlp import MLP

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
        self.hidden_dim_of_transformer = 512
        self.hidden_layers_of_transformer = 4
        
        self.input_projection = nn.Linear(self.enc_dim, self.hidden_dim_of_transformer)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim_of_transformer, nhead=8, dim_feedforward=2048)
        self.transformer_layer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.hidden_layers_of_transformer)
        self.output_projection = nn.Linear(self.hidden_dim_of_transformer, self.dec_dim)
        
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

        # Use Transformer layer for prediction
        x = self.input_projection(x_enc)
        x = self.transformer_layer(x) 
        predictions_flat = self.output_projection(x)

        state_vars = n_vars * self.ed_ratio
        # Reshape predictions back to original batch size and variable dimensions
        predictions = predictions_flat.view(bs, state_vars, -1).permute(0, 2, 1)  # [bs, seq_len, n_vars]

        return predictions.to(torch.float32)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)