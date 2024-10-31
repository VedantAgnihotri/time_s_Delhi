import torch
from torch import nn

class weatherGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
model = weatherGRU(4, 64, 4, 2).to(device='cuda')
state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)
model.eval()
print(model.parameters())