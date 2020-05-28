import xml.etree.ElementTree as ET 
import numpy as np 
import torch
from PIL import Image

def parseXML(path, name_to_label, c=0, is_pytorch=False):
    #Create a dictionnary from the XML file (given its path here) following naming patterns used by torchvision for detection
    #If is_pytorch is on, will load in tensors the values
    root = ET.parse(path).getroot()
    info = {}
    shape = (int(root.find('size/height').text),int(root.find('size/width').text),int(root.find('size/depth').text))
    folder = root.find('folder') .text
    name = root.find('filename').text
    info['boxes'] = []
    info['labels'] = []
    info['area'] = []
    info['image_id'] = np.array([c])
    for type_tag in root.findall('object'):
        info['labels'].append(name_to_label[type_tag.find('name').text])
        local_box = [int(type_tag.find('bndbox/xmin').text),int(type_tag.find('bndbox/ymin').text),int(type_tag.find('bndbox/xmax').text),int(type_tag.find('bndbox/ymax').text)]
        info['boxes'].append(local_box)
        info['area'].append((local_box[2]-local_box[0])*(local_box[3]-local_box[1]))
    info['boxes'] = np.array(info['boxes'])
    info['area'] = np.array(info['area'])
    info['labels'] = np.array(info['labels'])
    info['iscrowd'] = np.zeros_like(info['area'])

    if is_pytorch:
        info['boxes'] = torch.from_numpy(info['boxes']).type(torch.FloatTensor)
        info['area'] = torch.from_numpy(info['area']).type(torch.FloatTensor)
        info['labels'] = torch.from_numpy(info['labels']).type(torch.int64)
        info['image_id'] = torch.from_numpy(info['image_id']).type(torch.int64)
        info['iscrowd'] = torch.from_numpy(info['iscrowd']).type(torch.uint8)

    return name, folder, shape, info

def apply_bb_from_XML(image_name, xml_df, name_to_label, element_label):
    xml_path, label = xml_df.query('@image_name in path')
    im_path = im_df.query('@image_name in path')['path']
    _, _, _, info = parseXML(path, name_to_label)
    im = np.asarray(Image.open(im_path))

    for k in range(len(info['boxes'])):
        if label in element_label[info['labels'][k]]
            cv2.rectangle(im,info['boxes'][k][:2],info['boxes'][k][2:],(0,255,0),6)
        else :
            cv2.rectangle(im,info['boxes'][k][:2],info['boxes'][k][2:],(0,0,255),6)

    return im


