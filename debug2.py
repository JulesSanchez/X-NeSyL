import pandas as pd
from utils.parse_xml import parseXML
from utils.config import *
import cv2

path_xml = '/home/jules/Documents/Stage 4A/Data/Dataset-IGRB1092_14cls/files_xml.csv'
path_img = '/home/jules/Documents/Stage 4A/Data/Dataset-IGRB1092_14cls/files_img.csv'

xmls = pd.read_csv(path_xml)
imgs = pd.read_csv(path_img)
classes = [0,0,0,0]
part_label = [0 for i in range(len(SUB_ELEMENTS.keys()))]
index_parts = [0 for i in range(len(SUB_ELEMENTS.keys()))]
index_image = [0,0,0,0]

for i in range(len(xmls)):  
    xml_name, label = xmls.iloc[i]
    if label.endswith('man'):
        classes[0] += 1
        index_image[0] = i
    if label.endswith('ico'):
        classes[1] += 1
        index_image[1] = i
    if label.endswith('ta'):
        classes[2] += 1
        index_image[2] = i
    if label.endswith('oco'):
        classes[3] += 1
        index_image[3] = i
    name, folder, shape, info = parseXML(xml_name, SUB_ELEMENTS)
    for label in info['labels']:
        part_label[label-1] += 1
        index_parts[label-1] = i

print(index_parts)

for k in range(len(index_parts)):
    idx = index_parts[k]
    xml_name, label = xmls.iloc[idx]
    img_name, label = imgs.iloc[idx]
    name, folder, shape, info = parseXML(xml_name, SUB_ELEMENTS)
    img = cv2.imread(img_name)
    for j in range(len(info['labels'])):
        label = info['labels'][j]
        if label-1 == k:
            box = info['boxes'][j]
            cv2.imwrite(SUB_ELEMENTS_REVERSED[label] + '.png', img[int(box[1])-10:int(box[3])+10,int(box[0])-10:int(box[2])+10])


# import matplotlib.pyplot as plt
# plt.yticks(fontsize=19)
# plt.barh([i for i in range(len(classes))], classes, tick_label = ['hispanic-muslim', 'gothic', 'renaissance', 'baroque'])
# plt.show()