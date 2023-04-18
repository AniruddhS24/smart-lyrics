import torch
import torch.nn as nn

class SimpleLM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleLM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, input_size)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    # state is tuple of hidden and cell states
    def forward(self, x, state):
        embedded = self.embedding(x) # shape: seq_len x i_o_size
        lstm_output, (hidden_state_out, cell_state_out) = self.lstm(embedded, state)
        linear_output = self.linear(lstm_output)
        return linear_output, (hidden_state_out, cell_state_out)