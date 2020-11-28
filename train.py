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
from utils.knowledge_graph import compare_shap_and_KG, reduce_shap, GED_metric, get_bbox_weight
import utils.utils as uti

#Path to MonuMAI-AutomaticStyleClassification folder
sys.path.append("../MonuMAI-AutomaticStyleClassification")
from tools.pickle_tools import *

TMP_TRAIN = TMP_PATH + '/train'
TMP_VAL = TMP_PATH + '/val'
TMP_TEST = TMP_PATH + '/test'

os.makedirs(TMP_VAL, exist_ok=True)
os.makedirs(TMP_TRAIN, exist_ok=True)
os.makedirs(TMP_TEST, exist_ok=True)

##Argparse
parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
parser.add_argument('--resume', dest='resume', help='Whether or not to resume a training', default=False)
parser.add_argument('--path_resume', dest='path_resume', help='Path to the model to load', default='./model/model_monumenai.pth')
#path_resume also corresponds to where the file wille be saved in case of resume=False. If None default to DETECTOR_PATH

parser.add_argument('--epoch_classif', dest='epoch_classif', help='Number of epochs to train the classification model', default=150)
parser.add_argument('--batch_size', dest='batch_size', help='Batch size to train the classification model', default=64)
parser.add_argument('--neuron_classif', dest='neuron_classif', help='Number of neurons in the classification model', default=11)

parser.add_argument('--epoch_detection', dest='epoch_detection', help='Number of epochs to train the detection model', default=0)
#setting epoch_detection at 0 triggers the computation on the test set
parser.add_argument('--lr', dest='lr', help='Learning rate of the detection model', default=0.0003)
parser.add_argument('--stepLR', dest='stepLR', help='Step of the learning rate scheduler', default=9)
parser.add_argument('--gammaLR', dest='gammaLR', help='Gamma parameter of the learning rate scheduler', default=0.1)

parser.add_argument('--weight', dest='weight', help='Type of weighting', default='None')
#Type of weighting can be 'None", 'bbox_level', 'instance_level'
parser.add_argument('--exp_weights', dest='exp_weights', help='linear or exponential weighting', default='linear')
parser.add_argument('--data', dest='data', help='MonumenAI or PascalPart', default='MonumenAI')
args = parser.parse_args()

#Hyperparameters classification
n_neurons_classification = int(args.neuron_classif)
num_epochs_classification = int(args.epoch_classif)
batch_size_classification = int(args.batch_size)
learning_rate_classification = None

#Hyperparameters detection
num_epochs_detection = int(args.epoch_detection)
learning_rate_detection = float(args.lr)
stepLR = float(args.stepLR)
gammaLR = float(args.gammaLR)
data = args.data

if data == 'MonumenAI':
    from tools.metadata_tools import *
    from monumai.monument import Monument
    archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
    styles = FOLDERS_DATA
    #Loaders for detection
    train_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA, 'train.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
    if num_epochs_detection != 0:
        val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
    else:
        #Actually loading the test set
        val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
        test_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'test.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
if data == 'PascalPart':
    from tools.metadata_tools_pascal import *
    from monumai.pascal import Monument
    archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
    styles = list(PASCAL_EL_DIC.keys())
    #Loaders for detection
    train_loader = PascalDetectionDataset(train_pascal, PATH_PASCAL+PASCAL_IMG,PATH_PASCAL+PASCAL_XML, transform_detection_pascal)
    if num_epochs_detection != 0:
        val_loader = PascalDetectionDataset(val_pascal, PATH_PASCAL+PASCAL_IMG,PATH_PASCAL+PASCAL_XML, transform_detection_pascal)
    else:
        #Actually loading the test set
        val_loader = PascalDetectionDataset(val_pascal, PATH_PASCAL+PASCAL_IMG,PATH_PASCAL+PASCAL_XML, transform_detection_pascal)
        test_loader = PascalDetectionDataset(test_pascal, PATH_PASCAL+PASCAL_IMG,PATH_PASCAL+PASCAL_XML, transform_detection_pascal)

##Hyperparameters for detection
num_archi_features = len(archi_features)
num_classes_detection = num_archi_features + 1  # num_archi_features + background
num_styles = len(styles)


##Build detection model
if args.weight == "bbox_level":
    from utils.pytorch_utils import fasterrcnn_resnet50_fpn_custom
    detector = fasterrcnn_resnet50_fpn_custom(True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)
else:
    detector = models.detection.fasterrcnn_resnet50_fpn(True)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_detection)

if args.exp_weights == 'exponential':
    is_exponential = True
elif args.exp_weights == 'linear':
    is_exponential = False
else:
    print("Unrecognized type of weighting, defaulted to linear")
    is_exponential = False

detector.cuda()
#Optimizer
optimizer = torch.optim.SGD(detector.parameters(), lr=learning_rate_detection, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, stepLR, gamma=gammaLR, last_epoch=-1)

##Resume training if necessary
if args.resume:
    detector.load_state_dict(torch.load(args.path_resume))

##Loop ~ Epochs. First epoch is regular detection training
for j in range(num_epochs_detection+1):
    print('------------------')
    print("Epoch " + str(j+1) + '/' + str(num_epochs_detection))
    #Detection Hyperparameters
    if j > 0 or args.resume:
        detector.eval()
        #Run inference on all data to prepare for classification
        compute_json_detection(detector, train_loader, TMP_TRAIN,dataset=data)
        compute_json_detection(detector, val_loader, TMP_VAL,dataset=data)
        #If necessary run on test set aswell
        if num_epochs_detection == 0:
            compute_json_detection(detector, test_loader, TMP_TEST,dataset=data)

        print('Inference run')

        #Application of the aggregation function first for train then for val
        matrix_metadata = metadata_to_matrix(TMP_TRAIN, "json")
        names = matrix_metadata[:,-1]
        train_data = np.zeros((len(names),num_archi_features))
        train_label = np.zeros(len(names))
        if data == "MonumenAI":
            for i in range(len(names)):
                im_name = names[i][2:-4]
                idx = train_loader.images_loc['path'].str.contains(im_name)
                train_data[idx] = matrix_metadata[i,:num_archi_features]
                train_label[idx] = matrix_metadata[i,num_archi_features]
        if data == "PascalPart":
            for i in range(len(names)):
                im_name = os.path.join(PATH_PASCAL+PASCAL_IMG,names[i].split('_')[1][:-5] + '.jpg')
                idx = train_loader.images_loc.index(im_name)
                train_data[idx] = matrix_metadata[i,:num_archi_features]
                train_label[idx] = matrix_metadata[i,num_archi_features]
        train_data = train_data.astype(np.float32)
        train_label = to_categorical(train_label.astype(np.float32).astype(np.int8))

        matrix_metadata = metadata_to_matrix(TMP_VAL, "json")
        names = matrix_metadata[:,-1]
        test_data = np.zeros((len(names),num_archi_features))
        test_label = np.zeros(len(names))
        if data == "MonumenAI":
            for i in range(len(names)):
                im_name = names[i][2:-4]
                idx = val_loader.images_loc['path'].str.contains(im_name)
                test_data[idx] = matrix_metadata[i,:num_archi_features]
                test_label[idx] = matrix_metadata[i,num_archi_features]
        if data == "PascalPart":
            for i in range(len(names)):
                im_name = os.path.join(PATH_PASCAL+PASCAL_IMG,names[i].split('_')[1][:-5] + '.jpg')
                idx = val_loader.images_loc.index(im_name)
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
        #SHAP
        elements = np.random.choice(len(train_data), int(0.3*len(train_data)), False)
        explainer = shap.KernelExplainer(classificator.predict, train_data[elements])
        #Apply aggregation function to test if necessary
        if num_epochs_detection==0:
            matrix_metadata = metadata_to_matrix(TMP_TEST, "json")
            names = matrix_metadata[:,-1]
            test_data = np.zeros((len(names),num_archi_features))
            test_label = np.zeros(len(names))
            if data == "MonumenAI":
                for i in range(len(names)):
                    im_name = names[i][2:-4]
                    idx = test_loader.images_loc['path'].str.contains(im_name)
                    test_data[idx] = matrix_metadata[i,:num_archi_features]
                    test_label[idx] = matrix_metadata[i,num_archi_features]
            if data == "PascalPart":
                for i in range(len(names)):
                    im_name = os.path.join(PATH_PASCAL+PASCAL_IMG,names[i].split('_')[1][:-5] + '.jpg')
                    idx = test_loader.images_loc.index(im_name)
                    test_data[idx] = matrix_metadata[i,:num_archi_features]
                    test_label[idx] = matrix_metadata[i,num_archi_features]

            test_data = test_data.astype(np.float32)
            non_cat = np.copy(test_label)
            test_label = to_categorical(test_label.astype(np.float32).astype(np.int8))
        shap_values_test = explainer.shap_values(test_data, nsamples=30, l1_reg='bic')
        #Compute GED based on shap 
        d = GED_metric(test_data, shap_values_test, dataset=data)
        print('SHAP GED: ', d)
        if j < num_epochs_detection:
            #Compute relevant shap values & shap weights
            shap_values_train = explainer.shap_values(train_data, nsamples=30, l1_reg='bic')
            labels = np.argmax(train_label,axis=1)
            if args.weight == "instance_level":
                contributions_shap = compare_shap_and_KG(shap_values_train, labels, dataset=data)
                shap_coeff = reduce_shap(contributions_shap,is_exponential)
            elif args.weight == "bbox_level":
                shap_weights = get_bbox_weight(shap_values_train,is_exponential, dataset=data)
            print("Shap computed")
        print('Test loss: ', loss, '\tTest accuracy: ', accuracy)
    if j < num_epochs_detection:
        #Train detection
        metric_logger = uti.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', uti.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(j+1)

        index = 0
        detector.train()
        for images, targets in metric_logger.log_every(train_loader, 250, header):
            images = list(image.to('cuda') for image in images)
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
            #SHAP if necessary
            #-------
            if (j > 0 or args.resume) and args.weight == "bbox_level":
                loss_dict = detector(images, targets, weights=shap_weights[index][:,labels[index]])
                index += 1
            else :
                loss_dict = detector(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if (j > 0 or args.resume) and args.weight == "instance_level":
                losses = losses * shap_coeff[index]
                index += 1            
            #-------
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            metric_logger.update(loss=losses, **loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        scheduler.step()
        #Saving model
        if args.path_resume is None:
            torch.save(detector.state_dict(), DETECTOR_PATH)
        else:
            torch.save(detector.state_dict(), args.path_resume)
    #Evaluation
    if num_epochs_detection == 0:
        loss, accuracy = classificator.evaluate(test_data, test_label, verbose=1)
        predict_test = classificator(test_data)
        prediction = np.argmax(predict_test,axis=1)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(non_cat,prediction)
        STYLES_HOTONE_ENCODE = {'M' : 0, 'G' : 1, 'R' : 2, 'B' : 3}
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt 

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=STYLES_HOTONE_ENCODE.keys())
        disp.plot(include_values=True, cmap='viridis')
        plt.savefig('confmatDeLDECAS.png')
        shap_values_test = explainer.shap_values(test_data, nsamples=30, l1_reg='bic')
        #Compute GED based on shap 
        d = GED_metric(test_data, shap_values_test, dataset=data)
        print(d)
        print(accuracy)
    if j < num_epochs_detection or num_epochs_detection==0:
        evaluate(detector, test_loader, device="cuda")


shutil.rmtree(TMP_PATH)