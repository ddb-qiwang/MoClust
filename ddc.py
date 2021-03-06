# Define a ddc Layer

import torch.nn as nn

class none_layer(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.output = None
    
    def forward(self, x):
        self.output = x
        return self.output


class DDC(nn.Module):
    def __init__(self, input_dim, cfg):
     
        super().__init__()

        hidden_layers = [nn.Linear(input_dim[0], cfg.n_hidden), nn.ReLU()]
        if cfg.use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=cfg.n_hidden))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(cfg.n_hidden, cfg.n_clusters), nn.Softmax(dim=1))
        if cfg.direct:
            self.hidden = none_layer()
            self.output = nn.Sequential(nn.Linear(input_dim[0], cfg.n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden