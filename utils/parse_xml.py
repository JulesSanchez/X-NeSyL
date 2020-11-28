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
        info['parents'].append(type_tag.find('parts/ispartof'))
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
        if type_tag.find('parts/hasparts') is None or type_tag.find('parts/hasparts').text is None:
            for pt in type_tag.findall('polygon'):
                xs = pt.findall('pt/x')
                ys = pt.findall('pt/y')
            xs = [int(x.text) for x in xs]
            ys = [int(y.text) for y in ys]
            local_box = [min(xs),min(ys),max(xs),max(ys)]
            if local_box[3] > shape[0] or local_box[2] > shape[1] or local_box[2]<=local_box[0] or local_box[3]<=local_box[1]:
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

def getAllLabels(path,parts=False):
    all_names = set()
    for subpath in os.listdir(path):
        root = ET.parse(os.path.join(path,subpath)).getroot()
        for type_tag in root.findall('object'):
            if parts :
                if type_tag.find('parts/hasparts') is None:
                    all_names.add(type_tag.find('name').text)
            if not parts :
                if not type_tag.find('parts/hasparts') is None:
                    all_names.add(type_tag.find('name').text)
    return all_names

def setSingular(path):
    all_names = []
    for subpath in os.listdir(path):
        root = ET.parse(os.path.join(path,subpath)).getroot()
        l = 0
        for type_tag in root.findall('object'):
            if not type_tag.find('parts/hasparts') is None:
                l+= 1
        if l == 1:
            all_names.append(subpath)
    return all_names

def reconstruct_knowledge_base(path):
    PASCAL_EL_DIC = {'Bird': 0, 'Aeroplane': 1, 'Cat': 2, 'Dog': 3, 'Sheep': 4, 'Train': 5, 'Bicycle': 6, 'Horse': 7, 'Bottle': 8, 'Person': 9, 'Car': 10, 'diningtable': 11, 'Pottedplant': 12, 'Motorbike': 13, 'Sofa': 14, 'Boat': 15, 'Cow': 16, 'Chair': 17, 'Bus': 18, 'Tvmonitor': 19}
    PASCAL_PART_DIC = {'Arm': 0, 'Engine': 1, 'Coach': 2, 'Tail': 3, 'Pot': 4, 'Cap': 5, 'Ear': 6, 'Horn': 7, 'Ebrow': 8, 'Nose': 9, 'Torso': 10, 'Head': 11, 'Body': 12, 'Muzzle': 13, 'Beak': 14, 'Hand': 15, 'Hair': 16, 'Neck': 17, 'Foot': 18, 'Stern': 19, 'Artifact_Wing': 20, 'Locomotive': 21, 'License_plate': 22, 'Screen': 23, 'Mirror': 24, 'Saddle': 25, 'Hoof': 26, 'Door': 27, 'Leg': 28, 'Plant': 29, 'Mouth': 30, 'Animal_Wing': 31, 'Eye': 32, 'Chain_Wheel': 33, 'Bodywork': 34, 'Handlebar': 35, 'Headlight': 36, 'Wheel': 37, 'Window': 38}
    reverse_dic = {}
    out_dic = {}
    for key in PASCAL_EL_DIC:
        out_dic[key] = set()
        reverse_dic[PASCAL_EL_DIC[key]] = key
    for subpath in os.listdir(path):
        root = ET.parse(os.path.join(path,subpath)).getroot()
        children = {}
        parents = {}    
        for type_tag in root.findall('object'):
            if not type_tag.find('parts/hasparts') is None:
                parents[int(type_tag.find('id').text)] = type_tag.find('name').text
            else :
                children[int(type_tag.find('id').text)] = (type_tag.find('name').text,int(type_tag.find('parts/ispartof').text))
        for k in children:
            parent = parents[children[k][1]]
            out_dic[parent].add(children[k][0])
    for key in PASCAL_EL_DIC:
        out_dic[key] = list(out_dic[key])
    return out_dic

def get_label(path, name_to_label):
    root = ET.parse(path).getroot()                    
    for type_tag in root.findall('object'):
        if not type_tag.find('parts/hasparts') is None:
            label_name = type_tag.find('name').text
    return name_to_label[label_name]