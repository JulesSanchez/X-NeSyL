import torch 
import time, json, os
from .config import *
import numpy as np 
from .parse_xml import get_label

def train(model, dataloaders, criterion, optimizer):
    since = time.time()
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for i in range(len(dataloaders.images_loc)//dataloaders.batch_size + (len(dataloaders.images_loc)%dataloaders.batch_size > 0)):
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
    for i in range(len(dataloaders.images_loc)//dataloaders.batch_size + (len(dataloaders.images_loc)%dataloaders.batch_size > 0)):
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

def compute_json_detection(detector, loader, path, dataset = 'MonumenAI'):
    #Run inference on all data to prepare for classification
    detector.eval()
    if dataset == 'MonumenAI':
        information_about_class = ['M', 'G', 'R', 'B']
        elemen_dic = SUB_ELEMENTS
        reverse_dic = SUB_ELEMENTS_REVERSED
    if dataset == 'PascalPart':
        information_about_class = list(PASCAL_EL_DIC.keys())
        elemen_dic = PASCAL_PART_DIC
        reverse_dic = PASCAL_PART_DIC_REVERSED
    for k in range(len(loader)):

        img = loader[k][0][0].cuda()
        results = detector([img])[0]
        r_b = results['boxes'].detach().cpu().numpy()
        scores = results['scores'].detach().cpu().numpy()
        classes = results['labels'].detach().cpu().numpy()
        unique, counts = np.unique(classes, return_counts=True)
        counter = dict(zip(unique, counts))

        if dataset == 'MonumenAI':
            img_name = loader.images_loc.iloc[k, 0]
        if dataset == 'PascalPart':
            img_name = loader.images_loc[k]

        condensced_results = {}
        condensced_results["num_predictions"] = []
        condensced_results["image"] = img_name
        condensced_results["object"] = []
        if dataset == 'MonumenAI':
            condensced_results["true_label"] = int(loader.images_loc.iloc[k, 1])
        if dataset == 'PascalPart':
            condensced_results["true_label"] = get_label(loader.xml_loc[k], PASCAL_EL_DIC)

        for name in elemen_dic:

            if elemen_dic[name] in counter:
                condensced_results["num_predictions"].append({
                    name :  int(counter[elemen_dic[name]])
                })
            else:
                condensced_results["num_predictions"].append({
                    name :  0
                })
            
        for k in range(len(r_b)):
            box = r_b[k]/224.
            local_result = {
                "bndbox" : {
                    "xmin": str(box[0]),
                    "ymin": str(box[1]),
                    "ymax": str(box[3]),
                    "xmax": str(box[2])
                },
                "score" : str(scores[k]),
                "class" : reverse_dic[classes[k]]
            }
            condensced_results["object"].append(local_result)
        local_path = os.path.join(path,information_about_class[condensced_results["true_label"]] + '_' + img_name.split('/')[-1][:-4] + '.json')
        with open(local_path, 'w') as fp:
            json.dump(condensced_results, fp)