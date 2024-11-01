import time
import torch
import numpy as np


def train_one_epoch_teacher(model, dataloader, optimizer, loss_fn, device = 'cpu'):
    train_loss = [] # store the loss of every batch
    # operate every batch
    for i, data in enumerate(dataloader):
        inputs, labels = data
        # se fijan los gradientes en cero
        optimizer.zero_grad()
        # move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # make predictions
        outputs = model(inputs)
        # compute loss and gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # adjust weights 
        optimizer.step()
        # save batch loss
        train_loss.append(loss.detach().cpu().numpy())
    # return mean loss
    return np.mean(train_loss)


def get_val_loss_teacher(model, dataloader, loss_fn, device='cpu'):
    val_loss = []
    with torch.no_grad():
        # operate every batch
        for i, data in enumerate(dataloader):
            inputs, labels = data
            # move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # make predictions
            outputs = model(inputs.cuda())
            # compute loss
            loss = loss_fn(outputs, labels)
            # save batch loss
            val_loss.append(loss.detach().cpu().numpy())
    return np.mean(val_loss) # return mean loss


def get_acc(model, dataloader, device = 'cpu'):
    correct = 0
    total = 0
    with torch.no_grad():
        # operate every batch
        for i, data in enumerate(dataloader):
            inputs, labels = data
            # move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # make predictions
            outputs = model(inputs)
            outputs: torch.Tensor = torch.nn.functional.softmax(outputs, dim=1)
            outputs: torch.Tensor = torch.argmax(outputs, axis=1)
            # save correct predictions
            correct += (outputs == labels).float().sum().cpu()
            total += len(labels)
    # calculate accuracy
    accuracy = correct / total
    return accuracy.item()


def train_model_teacher(model,
                        trainloader,
                        valloader,
                        optimizer,
                        loss_fn,
                        epochs,
                        max_patience,
                        device = 'cpu',
                        model_path = None):
    
    # lists to store the evolution of the loss
    train_loss_evol = []
    val_loss_evol = []

    # lists to store the evolution of the accuracy
    train_acc_evol = []
    val_acc_evol = []

    # initialize validation acc
    best_vloss = np.inf

    begin_time = time.time() # initial time

    # training loop
    for epoch in range(epochs):
        print(f'EPOCH: {epoch + 1}')

        # one training iteration
        model.train()
        avg_loss = train_one_epoch_teacher(model, trainloader, optimizer, loss_fn, device)

        # evaluate the model in validation set
        model.eval()
        avg_vloss = get_val_loss_teacher(model, valloader, loss_fn, device)

        # save train and validation loss
        train_loss_evol.append(avg_loss)
        val_loss_evol.append(avg_vloss)
        
        # get and save accuracy in test and validation
        train_acc = get_acc(model, trainloader, device)
        val_acc = get_acc(model, valloader, device)
        train_acc_evol.append(train_acc)
        val_acc_evol.append(val_acc)
        
        # print the evolution of the metrics
        print(f'LOSS train {avg_loss} valid {avg_vloss} | ACC train {train_acc} val {val_acc}')
        
        # update the patience (number of epochs without improvement)
        patience += 1
        
        # save the best validation loss and save the model if necessary
        if avg_vloss < best_vloss:
            patience = 0
            best_vloss = avg_vloss
            if model_path:
                torch.save(model.state_dict(), model_path)
                print(f'Model saved in EPOCH {epoch+1}')
        
        # early stopping condition
        if patience == max_patience:
            break

    # save the training time
    end_time = time.time()
    execution_time = round(end_time - begin_time, 1)

    # save on a dictionary all the training metrics
    results = {'train_loss' : np.array(train_loss_evol).astype('float32'),
                'val_loss': np.array(val_loss_evol).astype('float32'),
                'train_acc': np.array(train_acc_evol).astype('float32'),
                'val_acc': np.array(val_acc_evol).astype('float32'),
                'execution_time': execution_time}
    
    # return the metrics
    return results