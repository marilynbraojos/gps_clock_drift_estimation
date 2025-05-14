import time
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error


def evaluate_model(model, test_dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    predictions = []
    actuals = []
    
    start_time = time.time()  # Record the start time

    
    with torch.no_grad():
        for X_test, y_test in test_dataloader:

            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            loss = criterion(outputs, y_test)
            running_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_test.cpu().numpy())
            
   
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f'Testing completed in {elapsed_time:.10e} seconds')

   
    test_loss = running_loss / len(test_dataloader)
    mae = mean_absolute_error(actuals, predictions)
    print(f'Test Loss (RMSE): {test_loss:.6e}, MAE: {mae:.6e}')

    return test_loss, mae, predictions, actuals