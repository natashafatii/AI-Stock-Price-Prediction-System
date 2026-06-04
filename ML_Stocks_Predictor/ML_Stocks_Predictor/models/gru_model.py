

import torch

import torch.nn as nn

class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) Neural Network Model.
    GRU is a type of Recurrent Neural Network (RNN) that is effective at capturing
    long-term dependencies in sequential data like stock prices. It is similar to LSTM
    but has a simpler architecture with fewer gates, making it faster to train 
    while often achieving similar predictive performance.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):

        super(GRUModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.gru = nn.GRU(

            input_dim, 

            hidden_dim, 

            num_layers, 

            batch_first=True, 

            dropout=dropout if num_layers > 1 else 0

        )

        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.gru(x, h0)

        out = out[:, -1, :]

        out = self.batch_norm(out)

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out

    def get_model_name(self):

        return "GRU"

