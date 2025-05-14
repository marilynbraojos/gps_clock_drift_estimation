import torch 
import torch.nn as nn

class RMSELoss(nn.Module): 
    def __init__(self):
        super().__init__()
        self.mse_fct = nn.MSELoss()

    def forward(self, yhat, y):
        mse_loss = self.mse_fct(yhat, y)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss
    
