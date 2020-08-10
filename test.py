##Imports
import numpy as np 
import os, sys, argparse, datetime, shutil
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json
from keras.utils import to_categorical
import shap 

#Import from current folder
from utils.config import *
from utils.dataloader import *
from utils.engine import train_one_epoch, evaluate
from utils.train import compute_json_detection
from utils.coco_utils import get_coco_api_from_dataset
from utils.coco_eval import CocoEvaluator
from utils.knowledge_graph import compare_shap_and_KG, reduce_shap
import utils.utils as uti

#Path to MonuMAI-AutomaticStyleClassification folder
sys.path.append("../MonuMAI-AutomaticStyleClassification")
from tools.pickle_tools import *
from tools.metadata_tools import *
from monumai.monument import Monument


archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
styles = FOLDERS_DATA

TMP_TRAIN = TMP_PATH + '/train'
TMP_VAL = TMP_PATH + '/val'

train_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA, 'train.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
matrix_metadata = metadata_to_matrix(TMP_TRAIN, "json")
names = matrix_metadata[:,-1]
train_data = np.zeros((len(names),14))
train_label = np.zeros(len(names))
for i in range(len(names)):
    im_name = names[i][1:-5]
    #im_name = img_name.split('/')[-1][:-4]
    idx = train_loader.images_loc['path'].str.contains(im_name)
    print(idx.iloc[0])
    train_data[idx] = matrix_metadata[i,:14]
    train_label[idx] = matrix_metadata[i,14]


train_data = train_data.astype(np.float32)
train_label = to_categorical(train_label.astype(np.float32).astype(np.int8))

