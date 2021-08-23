##Imports
import numpy as np 
import os, sys, argparse, datetime, shutil, json
from PIL import Image 
import cv2
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json
from keras.utils import to_categorical
import shap 
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

#Import from current folder
from utils.config import *
from utils.dataloader import *
from utils.engine import train_one_epoch, evaluate
from utils.train import compute_json_detection
from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
from utils.knowledge_graph import compare_shap_and_KG, reduce_shap, GED_metric, get_bbox_weight
import utils.utils as uti

#Path to MonuMAI-AutomaticStyleClassification folder
sys.path.append("../MonuMAI-AutomaticStyleClassification")
from tools.pickle_tools import *

##Argparse
parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
parser.add_argument('--path_resume', dest='path_resume', help='Path to the model to load', default='./model/model_fasterRCNN_noshap.pth')
parser.add_argument('--data', dest='data', help='MonumenAI or PascalPart', default='MonumenAI')
parser.add_argument('--path_image', dest='path_image', help='Either folder or image', default='/home/jules/Documents/Stage 4A/Data/')
parser.add_argument('--path_save', dest='path_save', help='Where to save results', default='./result')
args = parser.parse_args()

os.makedirs(args.path_save, exist_ok=True)

#Hyperparameters detection
data = args.data
if data == 'MonumenAI':
    from tools.metadata_tools import *
    from monumai.monument import Monument
    archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
    styles = FOLDERS_DATA
if data == 'PascalPart':
    from tools.metadata_tools_pascal import *
    from monumai.pascal import Monument
    archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
    styles = list(PASCAL_EL_DIC.keys())

num_archi_features = len(archi_features)
num_classes_detection = num_archi_features + 1  # num_archi_features + background
num_styles = len(styles)

##Build detection model
if "bbox" in args.path_resume:
    from utils.pytorch_utils import fasterrcnn_resnet50_fpn_custom
    detector = fasterrcnn_resnet50_fpn_custom(True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)
else:
    detector = models.detection.fasterrcnn_resnet50_fpn(True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)

detector.load_state_dict(torch.load(args.path_resume))

#Make loaders
from albumentations import (
    Resize,
    Compose,
    Normalize,
    CenterCrop
)
from albumentations.pytorch.transforms import ToTensor

transform =  Compose(
    [
        Resize(224,224),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToTensor()
    ],
    p=1
                                )


if args.path_image.endswith('.jpg'):
    #Resize the image and save ratio for upsampling later
    image = np.asarray(Image.open(args.path_image).convert('RGB'))
    ratio_h = image.shape[0]/224
    ratio_w = image.shape[1]/224
    padding_h = (256-224)/2 
    padding_w = (256-224)/2 
    augmented = transform(image=image)
    img = [augmented['image']]
    detector.eval()
    dic_results = detector(img)[0]
    boxes = []
    labels = []
    scores = []
    #Apply the boxes on the image
    for i in range(len(dic_results['boxes'])):
        xmin = (dic_results['boxes'][i][0].detach().numpy())*ratio_w
        ymin = (dic_results['boxes'][i][1].detach().numpy())*ratio_h
        xmax = (dic_results['boxes'][i][2].detach().numpy())*ratio_w
        ymax = (dic_results['boxes'][i][3].detach().numpy())*ratio_h
        boxes.append([xmin,ymin,xmax,ymax])
        labels.append(dic_results['labels'][i].item())
        scores.append(dic_results['scores'][i].item())
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),4)
        cv2.putText(image,archi_features[labels[i]-1],(int(xmin),int(ymin)-4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(image,str(scores[i])[:5],(int(xmin),int(ymax)-4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    #save the result
    result = Image.fromarray((image).astype(np.uint8))
    result.save(args.path_save + '/' + args.path_image.split('/')[-1])

else:
    #If it's a folder, explore the whole folder and apply the same as before
    results_dic = {}
    for path in os.listdir(args.path_image):
        if path.endswith('.jpg'):
            img_path = os.path.join(args.path_image,path)
            image = np.asarray(Image.open(img_path).convert('RGB'))
            ratio_h = image.shape[0]/224
            ratio_w = image.shape[1]/224
            padding_h = (256-224)/2 
            padding_w = (256-224)/2 
            augmented = transform(image=image)
            img = [augmented['image']]
            detector.eval()
            dic_results = detector(img)[0]
            boxes = []
            labels = []
            scores = []
            for i in range(len(dic_results['boxes'])):
                xmin = (dic_results['boxes'][i][0].detach().numpy())*ratio_w
                ymin = (dic_results['boxes'][i][1].detach().numpy())*ratio_h
                xmax = (dic_results['boxes'][i][2].detach().numpy())*ratio_w
                ymax = (dic_results['boxes'][i][3].detach().numpy())*ratio_h
                boxes.append([xmin,ymin,xmax,ymax])
                labels.append(dic_results['labels'][i].item())
                scores.append(dic_results['scores'][i].item())
                cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),4)
                cv2.putText(image,archi_features[labels[i]-1],(int(xmin),int(ymin)-4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(image,str(scores[i])[:5],(int(xmin),int(ymax)-4),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            #result = Image.fromarray((image).astype(np.uint8))
            #result.save(args.path_save + '/' + img_path.split('/')[-1])
            results_dic[path] = {'scores':scores, 'labels':labels}
            with open(args.path_save + '/' + os.path.basename(os.path.normpath(args.path_image)) + 'dic_res.json', 'w') as json_file:
                json.dump(results_dic, json_file)