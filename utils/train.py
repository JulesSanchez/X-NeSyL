import torch 
import time 

def train(model, dataloaders, criterion, optimizer):
    since = time.time()
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for i in range(len(dataloaders)):
        inputs, labels = dataloaders[i]
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders.images_loc)
    epoch_acc = running_corrects.double() / len(dataloaders.images_loc)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return 

def validate(model, dataloaders, criterion, best_acc, path):
    since = time.time()
    model.eval()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for i in range(len(dataloaders)):
        inputs, labels = dataloaders[i]
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        # deep copy the model
    epoch_loss = running_loss / len(dataloaders.images_loc)
    epoch_acc = running_corrects.double() / len(dataloaders.images_loc)
    if  epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), path)

    time_elapsed = time.time() - since
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return best_acc