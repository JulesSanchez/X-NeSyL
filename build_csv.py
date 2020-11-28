import os, torch
import pandas as pd
import torchvision.models as models
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils.config import *
from utils.dataloader import ArchitectureClassificationDataset, train_transform, val_transform, PascalClassificationDataset, transform_detection_pascal
from utils.train import train, validate

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

def split_train_val_csv(split=0.6):
    df_img = pd.read_csv(os.path.join(PATH_DATA,CSV_IMG))
    df_train, df_val = train_test_split(df_img, train_size=split, stratify=df_img['class'])
    df_val, df_test = train_test_split(df_val, train_size=0.5, stratify=df_val['class'])
    df_train.to_csv(os.path.join(PATH_DATA, 'train.csv'), index = False, header=True)
    df_val.to_csv(os.path.join(PATH_DATA, 'val.csv'), index = False, header=True)
    df_test.to_csv(os.path.join(PATH_DATA, 'test.csv'), index = False, header=True)


if not os.path.exists(os.path.join(PATH_DATA,CSV_IMG)):
    build_csv()

if not os.path.exists(os.path.join(PATH_DATA, 'test.csv')):
    split_train_val_csv()

    