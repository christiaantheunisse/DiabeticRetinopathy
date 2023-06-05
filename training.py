import os
import torch
from tqdm import tqdm
import torch.nn as nn

def train_one_epoch(model: nn.Module,
                    epoch_index: int,
                    training_loader: torch.utils.data.dataloader.DataLoader,
                    optimizer: torch.optim,
                    loss_fn: nn.Module,
                   ):
    """
    Run the training for one single epoch, so one loop over the whole training set
    
    Arguments:
        model (nn.Module): The PyTorch model
        epoch_index (int): the current epoch
        training_loader (torch.utils.data.dataloader.DataLoader): PyTorch data iterable that contains the data
        optimizer (torch.optim): the optimizer that updates the parameters
        loss_fn (nn.Module): the loss function
        
    Return:
        The average loss over the last batch
    """
    running_loss = 0.
    avg_loss = 0.

    for i, data in tqdm(enumerate(training_loader)):
        # Every data instance is an input + label pair
        device = next(model.parameters()).device
        
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
#         print(f"Model output: \n{outputs}")
#         print(f"labels: \n{labels}")
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        # Print an update every x batches
        update_batches = 50
        if i % update_batches == update_batches - 1:
            avg_loss = running_loss / update_batches # loss per batch
            print(f"  batch {i+1} loss: {avg_loss}")
            running_loss = 0.

    return avg_loss

def validate(model: nn.Module,
             epoch_index: int,
             validation_loader: torch.utils.data.dataloader.DataLoader,
             loss_fn: nn.Module = None,
             acc_fn: nn.Module = None,
            ):
    """
    Validate the model over the whole validation set
    
    Arguments:
        model (nn.Module): the PyTorch
        epoch_index (int): the current epoch
        validation_loader (torch.utils.data.dataloader.DataLoader): PyTorch data iterable that contains the data
        loss_fn (nn.Module): the loss function;
        acc_fn (nn.Module): the accuracy function
        
    Return:
        The average loss / validation score over the whole validation set
    """
    device = next(model.parameters()).device

    total_loss = 0 if loss_fn else None
    total_acc = 0 if acc_fn else None 

    with torch.no_grad():
        for i, data in tqdm(enumerate(validation_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if loss_fn:
                loss = loss_fn(outputs, labels)
                total_loss += loss
            if acc_fn:
                acc = acc_fn(outputs, labels)
                total_acc += acc
    
    avg_loss = total_loss / (i + 1) if loss_fn else None
    avg_acc = total_acc / (i + 1) if acc_fn else None
    
    print(f"The performance on the validation data after epoch {epoch_index} is:",
          f"\n\t average accuracy: {avg_acc}\n\t average loss: {avg_loss}")
    
    return avg_loss.item(), avg_acc.item()

def save_model(model: nn.Module, epoch_index: int, run_name: str, model_dir: str = "models"):
    """
    Function to save the model checkpoints to a directory with a specified name.
    
     Arguments:
        model (nn.Module): The PyTorch
        epoch_index (int): the current epoch
        run_name (str): Name the store the current model. Will also be the name of the directory it will be stored in
        model_dir (str): Directory where the models are stored
    """
    model_name = model.head.__class__.__name__
    model_name += "_" + str(epoch_index) + ".pth"
    
    if not os.path.exists(os.path.join(model_dir, run_name)):
        os.makedirs(os.path.join(model_dir, run_name))
    
    torch.save(model.head.state_dict(), os.path.join(model_dir, run_name, model_name))