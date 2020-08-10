##Imports
import numpy as np 
import os, sys, argparse, datetime, shutil
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
from utils.knowledge_graph import compare_shap_and_KG, reduce_shap, GED_metric
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

os.makedirs(TMP_VAL, exist_ok=True)
os.makedirs(TMP_TRAIN, exist_ok=True)

##Argparse
parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
parser.add_argument('--resume', dest='resume', help='Whether or not to resume a training', default=False)
parser.add_argument('--path_resume', dest='path_resume', help='Path to the model to load', default='./model/model_fasterRCNN_shap.pth')
parser.add_argument('--use_shap', dest='use_shap', help='Use shap during detection training', default=True)

parser.add_argument('--epoch_classif', dest='epoch_classif', help='Number of epochs to train the classification model', default=150)
parser.add_argument('--batch_size', dest='batch_size', help='Batch size to train the classification model', default=64)
parser.add_argument('--neuron_classif', dest='neuron_classif', help='Number of neurons in the classification model', default=11)

parser.add_argument('--epoch_detection', dest='epoch_detection', help='Number of epochs to train the detection model', default=10)
parser.add_argument('--lr', dest='lr', help='Learning rate of the detection model', default=0.003)
parser.add_argument('--stepLR', dest='stepLR', help='Step of the learning rate scheduler', default=4)
parser.add_argument('--gammaLR', dest='gammaLR', help='Gamma parameter of the learning rate scheduler', default=0.1)

args = parser.parse_args()

##Hyperparameters & Dataloaders
num_archi_features = len(archi_features)
num_classes_detection = 15  # num_archi_features + background
num_styles = len(styles)
#Hyperparameters classification
n_neurons_classification = args.neuron_classif
num_epochs_classification = args.epoch_classif
batch_size_classification = args.batch_size
learning_rate_classification = None
#Hyperparameters detection
num_epochs_detection = args.epoch_detection
learning_rate_detection = args.lr
stepLR = args.stepLR
gammaLR = args.gammaLR
#Loaders for detection
train_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA, 'train.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)

##Build detection model
detector = models.detection.fasterrcnn_resnet50_fpn(True)
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)
detector.cuda()
#Optimizer
optimizer = torch.optim.SGD(detector.parameters(), lr=learning_rate_detection, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepLR, gamma=gammaLR, last_epoch=-1)

##Resume training if necessary
if args.resume:
    detector.load_state_dict(torch.load(args.path_resume))

##Loop ~ Epochs. First epoch is regular detection training
for i in range(num_epochs_detection):
    print('------------------')
    print("Epoch " + str(i+1) + '/' + str(num_epochs_detection))
    #Detection Hyperparameters

    #Run inference on all data to prepare for classification
    compute_json_detection(detector, train_loader, TMP_TRAIN)
    compute_json_detection(detector, val_loader, TMP_VAL)

    print('Inference run')

    #Featurization
    matrix_metadata = metadata_to_matrix(TMP_TRAIN, "json")
    names = matrix_metadata[:,-1]
    train_data = np.zeros((len(names),num_archi_features))
    train_label = np.zeros(len(names))
    for i in range(len(names)):
        im_name = names[i][1:-5]
        idx = train_loader.images_loc['path'].str.contains(im_name)
        train_data[idx] = matrix_metadata[i,:num_archi_features]
        train_label[idx] = matrix_metadata[i,num_archi_features]


    train_data = train_data.astype(np.float32)
    train_label = to_categorical(train_label.astype(np.float32).astype(np.int8))

    matrix_metadata = metadata_to_matrix(TMP_VAL, "json")
    names = matrix_metadata[:,-1]
    test_data = np.zeros((len(names),num_archi_features))
    test_label = np.zeros(len(names))
    for i in range(len(names)):
        im_name = names[i][1:-5]
        idx = val_loader.images_loc['path'].str.contains(im_name)
        test_data[idx] = matrix_metadata[i,:num_archi_features]
        test_label[idx] = matrix_metadata[i,num_archi_features]


    test_data = test_data.astype(np.float32)
    test_label = to_categorical(test_label.astype(np.float32).astype(np.int8))

    #Classification training
    #Initialize model
    classificator = keras.Sequential()
    classificator.add(keras.layers.Dense(units=n_neurons_classification, activation='relu', input_shape=(num_archi_features,)))
    classificator.add(keras.layers.Dense(units=num_styles, activation='softmax'))
    classificator.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #Train model
    history = classificator.fit(train_data, train_label, batch_size=batch_size_classification, epochs=num_epochs_classification, verbose=0)
    loss, accuracy = classificator.evaluate(test_data, test_label, verbose=1)
    print('Test loss: ', loss, '\tTest accuracy: ', accuracy)

    #Shap
    elements = np.random.choice(len(train_data), int(0.3*len(train_data)), False)
    explainer = shap.KernelExplainer(classificator.predict, train_data[elements])
    shap_values_train = explainer.shap_values(train_data, nsamples=20, l1_reg="auto")
    shap_values_test = explainer.shap_values(test_data, nsamples=20, l1_reg="auto")
    #Compute GED based on shap metrics ?
    d = GED_metric(shap_values_test)
    print('SHAP GED: ', d)
    #Compute relevant shap values
    #labels = np.argmax(train_label,axis=1)
    labels = np.argmax(classificator(train_data).numpy(),axis=1)
    contributions_shap = compare_shap_and_KG(shap_values_train, labels)
    shap_coeff = reduce_shap(contributions_shap)
    print("Shap computed")



    #Train detection
    detector.train()
    metric_logger = uti.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', uti.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(i+1)

    index = 0

    for images, targets in metric_logger.log_every(train_loader, 250, header):
        images = list(image.to('cuda') for image in images)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
        loss_dict = detector(images, targets)
        #SHAP can be added during this computation
        #-------
        losses = sum(loss for loss in loss_dict.values())
        if args.use_shap and (i > 0 or args.resume):
            losses = losses * shap_coeff[index]
            index += 1            
        #-------
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    #Evaluation
    evaluate(detector, val_loader, device="cuda")
    scheduler.step()
    torch.save(detector.state_dict(), DETECTOR_PATH)

shutil.rmtree(TMP_PATH)
##Save model
torch.save(detector.state_dict(), DETECTOR_PATH)