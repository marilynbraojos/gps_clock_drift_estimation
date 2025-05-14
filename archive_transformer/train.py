import time
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error

def train_model(model, d_model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    val_losses = []
    train_mae = []
    val_mae = []

    start_time = time.time()  # Record the start time
    # input_projection = nn.Linear(1, d_model) 
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_actuals = []
        
        for X_train, y_train in train_dataloader:

            # X_train = input_projection(X_train.float())
            # y_train = input_projection(y_train.float())

            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            outputs = model(X_train, y_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_train_preds.extend(outputs.detach().cpu().numpy())
            all_train_actuals.extend(y_train.detach().cpu().numpy())
        
        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)
        train_mae_val = mean_absolute_error(all_train_actuals, all_train_preds)
        train_mae.append(train_mae_val)
        
        model.eval()
        running_loss = 0.0
        all_val_preds = []
        all_val_actuals = []
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                # X_val = input_projection(X_val.float())
                # y_val = input_projection(y_val.float())

                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val, y_val)
                loss = criterion(outputs, y_val)
                running_loss += loss.item()
                
                all_val_preds.extend(outputs.detach().cpu().numpy())
                all_val_actuals.extend(y_val.detach().cpu().numpy())
        
        val_loss = running_loss / len(val_dataloader)
        val_losses.append(val_loss)
        val_mae_val = mean_absolute_error(all_val_actuals, all_val_preds)
        val_mae.append(val_mae_val)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}')
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f'Training and validation completed in {elapsed_time:.10e} seconds')

    return train_losses, train_mae, val_losses, val_mae