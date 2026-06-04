

import torch

import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) Neural Network Model.
    LSTM is an advanced type of Recurrent Neural Network (RNN) specifically designed
    to learn long-term dependencies. It uses a system of gates (input, output, forget)
    to remember important information over long time sequences, making it highly suitable
    for time-series forecasting like stock price predictions.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):

        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.lstm = nn.LSTM(

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

        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.batch_norm(out)

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out

    def get_model_name(self):

        return "LSTM"

