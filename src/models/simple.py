import torch
import torch.nn as nn

class SimpleLM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(SimpleLM, self).__init__()
        self.kwargs = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hidden_size
        }
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, input_size)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.LazyLinear(output_size)
        
    # state is tuple of hidden and cell states
    def forward(self, x, state):
        embedded = self.embedding(x) # torch.Size([64, 512, 64])
        # print("embedded shape:", embedded.shape)
        lstm_output, (hidden_state_out, cell_state_out) = self.lstm(embedded, state) 
        # print("lstm shape:", lstm_output.shape) # lstm shape: torch.Size([64, 512, 128])
        linear_output = self.linear(lstm_output) # linear_output shape: torch.Size([64, 512, 30522])
        # print("output size: ", self.output_size)
        # print("linear_output shape:", linear_output.shape)
        return linear_output, (hidden_state_out, cell_state_out)

    # TODO finish this method   
    def sample(self):
        pass