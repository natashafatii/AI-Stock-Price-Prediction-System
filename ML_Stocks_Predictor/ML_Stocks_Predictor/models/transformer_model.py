

import torch

import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the data points in the sequence.
    Since Transformer models process all data simultaneously rather than sequentially (like RNNs),
    they need this encoding to understand the chronological order of the time-series stock data.
    """

    def __init__(self, d_model, max_len=5000):

        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """
    Transformer Neural Network Model.
    Unlike LSTMs or GRUs, Transformers use self-attention mechanisms to weigh the importance
    of different parts of the input sequence simultaneously. This allows them to capture complex
    patterns across time very effectively. Originally built for natural language processing, 
    they are now powerful tools for time-series forecasting.
    """

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=1, dropout=0.2):

        super(TransformerModel, self).__init__()

        self.d_model = d_model

        self.input_dim = input_dim

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=d_model * 4,

            dropout=dropout,

            batch_first=True

        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.fc1 = nn.Linear(d_model, d_model // 2)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(d_model // 2, output_dim)

    def forward(self, x):

        x = self.input_projection(x)

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = x[:, -1, :]

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

    def get_model_name(self):

        return "Transformer"

