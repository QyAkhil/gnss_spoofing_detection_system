import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2):
        super().__init__()
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.1)
        self.enc_fc  = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec_fc  = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=0.1)
        self.out_fc  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        seq_len     = x.size(1)
        enc_out, _  = self.encoder(x)
        z           = self.enc_fc(enc_out[:, -1, :])
        h           = self.dec_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _  = self.decoder(h)
        return self.out_fc(dec_out)