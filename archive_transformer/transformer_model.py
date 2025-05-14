import torch 
import torch.nn as nn 

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model = 256):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)
        
        self.output_layer = nn.Linear(d_model, output_dim)
    def forward(self, X, y=None):
        X = self.embedding(X)

        if y is not None: 
            y = self.embedding (y) 
        
        output = self.transformer(X,y)
        output = self.output_layer(output)
        
        return output