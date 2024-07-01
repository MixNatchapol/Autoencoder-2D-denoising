import torch
import torch.nn as nn

class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, latent_dim=128, num_layers=4, dropout_prob=0.3):
        super(RecurrentAutoencoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.encoder_conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.encoder_bn1 = nn.BatchNorm1d(64)
        self.encoder_bn2 = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.encoder_lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_dim * 2, latent_dim)  # *2 because of bidirectional

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim * 2)  # *2 because of bidirectional
        self.decoder_lstm = nn.LSTM(hidden_dim * 2, hidden_dim * 2, num_layers, batch_first=True, bidirectional=True, dropout=dropout_prob)

        self.decoder_conv1 = nn.Conv1d(hidden_dim * 2 * 2, 128, kernel_size=3, stride=1, padding=1)  # *2 because of bidirectional
        self.decoder_conv2 = nn.Conv1d(128, input_dim, kernel_size=3, stride=1, padding=1)
        self.decoder_bn1 = nn.BatchNorm1d(128)
        self.decoder_bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        # LSTM expects input shape (batch_size, sequence_length, input_dim)
        x = x.squeeze(1).transpose(1, 2)  # From (batch_size, 1, n_mels, time) to (batch_size, time, n_mels)
        
        # Encoder Convolutional Layers
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.leaky_relu(x)
        x = self.encoder_conv2(x)
        x = self.encoder_bn2(x)
        x = self.leaky_relu(x)
        
        seq_len = x.size(2)  # Get the sequence length after convolutions
        x = x.transpose(1, 2)  # From (batch_size, feature_maps, seq_len) to (batch_size, seq_len, feature_maps)

        # Encoder LSTM
        x, _ = self.encoder_lstm(x)
        x = self.fc1(x[:, -1, :])  # Take the last time step output and pass it through the fully connected layer

        # Prepare hidden and cell states for the decoder
        hidden = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim * 2).to(x.device)  # 2 for bidirectional
        cell = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim * 2).to(x.device)

        # Decoder
        x = self.fc2(x).unsqueeze(1).repeat(1, seq_len, 1)  # Repeat for each time step
        x, _ = self.decoder_lstm(x, (hidden, cell))
        x = x.transpose(1, 2)  # From (batch_size, seq_len, hidden_dim*2*2) to (batch_size, hidden_dim*2*2, seq_len)

        # Decoder Convolutional Layers
        x = self.decoder_conv1(x)
        x = self.decoder_bn1(x)
        x = self.leaky_relu(x)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = self.leaky_relu(x)

        x = x.transpose(1, 2).unsqueeze(1)  # Back to (batch_size, 1, n_mels, time)
        return x

# Ensure this file is saved as model.py to work with the rest of your setup
