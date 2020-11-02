from utils.config import *
from utils.parse_xml import parseXML
import pandas as pd
import os 
import csv 

train = os.path.join(PATH_DATA, 'train.csv')
test = os.path.join(PATH_DATA, 'test.csv')
val = os.path.join(PATH_DATA,'val.csv')
xml = os.path.join(PATH_DATA, CSV_XML)

out_train = []
out_val = []
out_class = []
out_test = []

s_e = SUB_ELEMENTS

for k in list(s_e.keys()):
    out_class.append([k,s_e[k]-1])

s_e_rev = SUB_ELEMENTS_REVERSED
im_names = pd.read_csv(train)
xml_loc =  pd.read_csv(xml)
for i in range(len(im_names)):
    im_name = im_names.iloc[i, 0]
    im_name_reduced = im_name.split('/')[-1][:-4]
    xml_name, label = xml_loc[xml_loc['path'].str.contains(im_name_reduced)].values[0]
    name, folder, shape, target = parseXML(xml_name, s_e, i, False)
    for k in range(len(target['boxes'])):
        out_train.append([im_name,target['boxes'][k][0],target['boxes'][k][1],target['boxes'][k][2],target['boxes'][k][3],s_e_rev[target['labels'][k]]])

print(len(out_train))

s_e_rev = SUB_ELEMENTS_REVERSED
im_names = pd.read_csv(val)
xml_loc =  pd.read_csv(xml)
for i in range(len(im_names)):
    im_name = im_names.iloc[i, 0]
    im_name_reduced = im_name.split('/')[-1][:-4]
    xml_name, label = xml_loc[xml_loc['path'].str.contains(im_name_reduced)].values[0]
    name, folder, shape, target = parseXML(xml_name, s_e, i, False)
    for k in range(len(target['boxes'])):
        out_val.append([im_name,target['boxes'][k][0],target['boxes'][k][1],target['boxes'][k][2],target['boxes'][k][3],s_e_rev[target['labels'][k]]])

im_names = pd.read_csv(test)
xml_loc =  pd.read_csv(xml)
for i in range(len(im_names)):
    im_name = im_names.iloc[i, 0]
    im_name_reduced = im_name.split('/')[-1][:-4]
    xml_name, label = xml_loc[xml_loc['path'].str.contains(im_name_reduced)].values[0]
    name, folder, shape, target = parseXML(xml_name, s_e, i, False)
    for k in range(len(target['boxes'])):
        out_test.append([im_name,target['boxes'][k][0],target['boxes'][k][1],target['boxes'][k][2],target['boxes'][k][3],s_e_rev[target['labels'][k]]])

with open('train_retinanet.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for row in out_train:
        wr.writerow(row)

with open('val_retinanet.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for row in out_val:
        wr.writerow(row)

with open('test_retinanet.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for row in out_test:
        wr.writerow(row)

with open('class_retinanet.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     for row in out_class:
        wr.writerow(row)