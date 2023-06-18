import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(model, model_type, epochs, loss_fn, patience, learning_rate, train_loader, 
          val_loader, shapes, l1=False, l2=False, ElNet=False):
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # every 20 steps - fixed, no changes
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)

    min_val_loss = 9999
    counter = 0
    l1_weight = 0.01
    l2_weight = 0.01
    for i in range(epochs):
        mse_train = 0
        for batch_x, batch_y in train_loader :
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            opt.zero_grad()
            y_pred = model(batch_x)
            y_pred = y_pred.squeeze(1)
            l = loss_fn(y_pred, batch_y)

            # Using l1 regularization
            if l1:
                l1_parameters = []
                for parameter in model.parameters():
                    l1_parameters.append(parameter.view(-1))
                L1 = l1_weight * model.compute_l1(torch.cat(l1_parameters))
                l += L1

            # Using l2 regularization
            if l2:
                l2_parameters = []
                for parameter in model.parameters():
                    l2_parameters.append(parameter.view(-1))
                L1 = l2_weight * model.compute_l2(torch.cat(l2_parameters))
                l += L1  

            if ElNet:
                # Specify L1 and L2 weights
                l1_weight = 0.8
                l2_weight = 0.2
                
                # Compute L1 and L2 loss component
                parameters = []
                for parameter in model.parameters():
                    parameters.append(parameter.view(-1))
                L1 = l1_weight * model.compute_l1(torch.cat(parameters))
                L2 = l2_weight * model.compute_l2(torch.cat(parameters))
                
                # Add L1 and L2 loss components
                l += L1
                l += L2      

            l.backward()
            mse_train += l.item()*batch_x.shape[0]
            opt.step()
        epoch_scheduler.step()
        with torch.no_grad():
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output = model(batch_x)
                output = output.squeeze(1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                mse_val += loss_fn(output, batch_y).item()*batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        
        if min_val_loss > mse_val**0.5:
            min_val_loss = mse_val**0.5
            print("Saving...")
            if model_type in ["RNN", "LSTM", "GRU"]:
                if l1:
                    torch.save(model.state_dict(), f"ConvRNN/conv{model_type}_l1_sml2010.pt")
                if l2:
                    torch.save(model.state_dict(), f"ConvRNN/conv{model_type}_l2_sml2010.pt") 
                if ElNet:
                    torch.save(model.state_dict(), f"ConvRNN/conv{model_type}_elnet_sml2010.pt") 
                else:
                    torch.save(model.state_dict(), f"ConvRNN/conv{model_type}_sml2010.pt")               
            counter = 0
        else: 
            counter += 1
        
        if counter == patience:
            break
        print("Iter: ", i, "train: ", (mse_train/shapes[0][0])**0.5, "val: ", (mse_val/shapes[1][0])**0.5)


def evaluate(test_loader, model, loss_fn, y_train_max, y_train_min):
    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = model(batch_x)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss_fn(output, batch_y).item()*batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)

        # resizing to true value scales
        preds = preds*(y_train_max - y_train_min) + y_train_min
        true = true*(y_train_max - y_train_min) + y_train_min

        mse = mean_squared_error(true, preds)
        mae = mean_absolute_error(true, preds)

        print(mse, mae)

        plt.figure(figsize=(20, 10))
        plt.plot(preds, 'b', label='line 1')
        plt.plot(true, 'g',  label='line 2')
        plt.show()