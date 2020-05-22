from config import *
import os, torch
import pandas as pd
import torchvision.models as models
from dataloader import ArchitectureClassificationDataset, train_transform, val_transform
from sklearn.model_selection import train_test_split
from train import train, validate

def build_csv():
    row_list_xml = []
    row_list_img = []
    label = 0
    for fold in FOLDERS_DATA:
        local_path = os.path.join(os.path.join(PATH_DATA, fold), 'xml')
        for el in os.listdir(local_path):
            row_list_xml.append({'path': os.path.join(local_path,el), 'class' : fold})
            row_list_img.append({'path': os.path.join(local_path[:-4],el[:-4] +'.jpg'), 'class' : label})
        label += 1
    df_xml = pd.DataFrame(row_list_xml)               
    df_img = pd.DataFrame(row_list_img)               
    df_xml.to_csv(os.path.join(PATH_DATA,CSV_XML), index = False, header=True)
    df_img.to_csv(os.path.join(PATH_DATA,CSV_IMG), index = False, header=True)

def split_train_val_csv(split=0.66):
    df_img = pd.read_csv(os.path.join(PATH_DATA,CSV_IMG))
    df_train, df_val = train_test_split(df_img, train_size=split, stratify=df_img['class'])
    df_train.to_csv(os.path.join(PATH_DATA, 'train.csv'), index = False, header=True)
    df_val.to_csv(os.path.join(PATH_DATA, 'val.csv'), index = False, header=True)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__ == '__main__':

    if not os.path.exists(os.path.join(PATH_DATA,CSV_IMG)):
        build_csv()

    if not os.path.exists(os.path.join(PATH_DATA, 'train.csv')):
        split_train_val_csv()

    model = models.resnet101(pretrained=True,num_classes=1000)
    set_parameter_requires_grad(model, True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048,512),
        torch.nn.ReLU(),
        torch.nn.Linear(512,4)
    )

    model.cuda()

    train_loader = ArchitectureClassificationDataset(os.path.join(PATH_DATA, 'train.csv'), BATCH_SIZE, train_transform)
    val_loader = ArchitectureClassificationDataset(os.path.join(PATH_DATA,'val.csv'), BATCH_SIZE, val_transform)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.8, last_epoch=-1)

    best_acc = 0

    for i in range(N_EPOCHS):
        print('Epoch {}/{}'.format(i+1, N_EPOCHS))
        print('-' * 10)
        train(model, train_loader, criterion, optimizer)
        best_acc = validate(model, val_loader, criterion, best_acc, MODEL_PATH)
        scheduler.step()