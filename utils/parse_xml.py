import xml.etree.ElementTree as ET 
import numpy as np 
import torch
from PIL import Image
import cv2, os

def parseXML(path, name_to_label, c=0, is_pytorch=False):
    #Create a dictionnary from the XML file (given its path here) following naming patterns used by torchvision for detection
    #If is_pytorch is on, will load in tensors the values
    root = ET.parse(path).getroot()
    info = {}
    shape = (int(root.find('size/height').text),int(root.find('size/width').text),3)
    folder = root.find('folder') .text
    name = root.find('filename').text
    info['boxes'] = []
    info['labels'] = []
    info['area'] = []
    info['image_id'] = np.array([c])
    for type_tag in root.findall('object'):
        local_box = [int(type_tag.find('bndbox/xmin').text),int(type_tag.find('bndbox/ymin').text),int(type_tag.find('bndbox/xmax').text),int(type_tag.find('bndbox/ymax').text)]
        if local_box[3] > shape[0] or local_box[2] > shape[1]:
            continue
        info['labels'].append(name_to_label[type_tag.find('name').text])
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

def apply_bb_from_XML(image_name, xml_df, im_df, name_to_label, element_label, heatmap = None):
    xml_path, label = xml_df[xml_df['path'].str.contains(image_name)].values[0]
    im_path, _ = im_df[im_df['path'].str.contains(image_name)].values[0]
    _, _, _, info = parseXML(xml_path, name_to_label)
    im = np.asarray(Image.open(im_path))
    if heatmap is not None:
        from skimage.transform import resize 
        try:
            heatmap = resize(heatmap,im.shape[:2],clip=False,preserve_range=True)
        except:
            heatmap = resize(heatmap,im.shape,clip=False,preserve_range=True)
    for k in range(len(info['boxes'])):
        if label in element_label[info['labels'][k]]:
            if heatmap is not None:
                cv2.rectangle(heatmap,(info['boxes'][k][0],info['boxes'][k][1]),(info['boxes'][k][2],info['boxes'][k][3]),(0,255,0),6)
            cv2.rectangle(im,(info['boxes'][k][0],info['boxes'][k][1]),(info['boxes'][k][2],info['boxes'][k][3]),(0,255,0),6)
        else :
            if heatmap is not None:
                cv2.rectangle(heatmap,(info['boxes'][k][0],info['boxes'][k][1]),(info['boxes'][k][2],info['boxes'][k][3]),(0,0,255),6)
            cv2.rectangle(im,(info['boxes'][k][0],info['boxes'][k][1]),(info['boxes'][k][2],info['boxes'][k][3]),(0,0,255),6)

    return im, heatmap

def readXML_pascal(path, name_to_label, c=0, is_pytorch=False):
    #Create a dictionnary from the XML file (given its path here) following naming patterns used by torchvision for detection
    #If is_pytorch is on, will load in tensors the values
    root = ET.parse(path).getroot()
    info = {}
    shape = (int(root.find('imagesize/nrows').text),int(root.find('imagesize/ncols').text),3)
    folder = root.find('folder').text
    name = root.find('filename').text
    info['boxes'] = []
    info['labels'] = []
    info['area'] = []
    info['image_id'] = np.array([c])
    info['parents'] = []
    for type_tag in root.findall('object'):
        info['parents'].append(type_tag.find('parts/ispartsof'))
        for pt in type_tag.findall('polygon'):
            xs = pt.findall('pt/x')
            ys = pt.findall('pt/y')
        xs = [int(x.text) for x in xs]
        ys = [int(y.text) for y in ys]
        local_box = [min(xs),min(ys),max(xs),max(ys)]
        if local_box[3] > shape[0] or local_box[2] > shape[1]:
            continue
        info['labels'].append(name_to_label[type_tag.find('name').text])
        info['boxes'].append(local_box)
        info['area'].append((local_box[2]-local_box[0])*(local_box[3]-local_box[1]))
    info['boxes'] = np.array(info['boxes'])
    info['area'] = np.array(info['area'])
    info['labels'] = np.array(info['labels'])
    info['iscrowd'] = np.zeros_like(info['area'])
    info['parents'] = np.arrays(info['parents'])
    return name, folder, shape, info

def parseXML_pascal(path, name_to_label, c=0, is_pytorch=False):
    #Create a dictionnary from the XML file (given its path here) following naming patterns used by torchvision for detection
    #If is_pytorch is on, will load in tensors the values
    root = ET.parse(path).getroot()
    info = {}
    shape = (int(root.find('imagesize/nrows').text),int(root.find('imagesize/ncols').text),3)
    folder = root.find('folder').text
    name = root.find('filename').text
    info['boxes'] = []
    info['labels'] = []
    info['area'] = []
    info['image_id'] = np.array([c])
    for type_tag in root.findall('object'):
        if type_tag.find('parts/hasparts') is None:
            for pt in type_tag.findall('polygon'):
                xs = pt.findall('pt/x')
                ys = pt.findall('pt/y')
            xs = [int(x.text) for x in xs]
            ys = [int(y.text) for y in ys]
            local_box = [min(xs),min(ys),max(xs),max(ys)]
            if local_box[3] > shape[0] or local_box[2] > shape[1]:
                continue
            info['labels'].append(name_to_label[type_tag.find('name').text])
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

def getAllLabels(path):
    all_names = set()
    for subpath in os.listdir(path):
        root = ET.parse(os.path.join(path,subpath)).getroot()
        for type_tag in root.findall('object'):
            all_names.add(type_tag.find('name').text)
    return all_names

#readXML_pascal('/home/jules/Documents/Stage 4A/Data/pascalPartDataset/AnnotationsPascalPart/000002.xml',{})
sets = getAllLabels('/home/jules/Documents/Stage 4A/Data/pascalPartDataset/AnnotationsPascalPart')
element_dic = {}
i = 0
for el in sets:
    element_dic[el] = i
    i += 1
print(element_dic)