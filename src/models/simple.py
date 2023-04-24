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
        x = x.long()
        embedded = self.embedding(x) # torch.Size([64, 512, 64])
        # print("embedded shape:", embedded.shape)
        lstm_output, (hidden_state_out, cell_state_out) = self.lstm(embedded, state) 
        # print("lstm shape:", lstm_output.shape) # lstm shape: torch.Size([64, 512, 128])
        linear_output = self.linear(lstm_output) # linear_output shape: torch.Size([64, 512, 30522])
        # print("output size: ", self.output_size)
        # print("linear_output shape:", linear_output.shape)
        return linear_output, (hidden_state_out, cell_state_out)
 
    def sample(self, length):
        h = torch.zeros((1, self.hidden_size))
        c = torch.zeros((1, self.hidden_size))
        context_arr = [0]
        context = torch.tensor(context_arr)
        for i in range(length):
            len_context = len(context)
            outputs, (h, c) = self(context, (h.detach(), c.detach()))
            next_best_index = torch.multinomial(outputs[len_context-1].softmax(dim=0), 1)
            # update context
            context_arr.append(next_best_index.item())
            context = torch.tensor(context_arr)
        return context[1:]