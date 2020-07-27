import os, cv2
import torchvision.models as models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.config import *
from utils.dataloader import *
from utils.engine import train_one_epoch, evaluate
import numpy as np
import json

if __name__ == '__main__':
    detector = models.detection.fasterrcnn_resnet50_fpn(True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 15  # 14 class (architectural elements) + background
    # get number of input features for the classifier
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    detector.cuda()
    detector.load_state_dict(torch.load(DETECTOR_PATH))
    detector.eval()

    val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
    train_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA, 'train.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)

    information_about_class = ['M', 'G', 'R', 'B']

    for k in range(len(train_loader)):

        img = train_loader[k][0][0].cuda()
        results = detector([img])[0]
        r_b = results['boxes'].detach().cpu().numpy()
        scores = results['scores'].detach().cpu().numpy()
        classes = results['labels'].detach().cpu().numpy()
        unique, counts = np.unique(classes, return_counts=True)
        counter = dict(zip(unique, counts))

        img_name = train_loader.images_loc.iloc[k, 0]

        condensced_results = {}
        condensced_results["num_predictions"] = []
        condensced_results["image"] = img_name
        condensced_results["object"] = []
        condensced_results["true_label"] = int(train_loader.images_loc.iloc[k, 1])

        for name in SUB_ELEMENTS:

            if SUB_ELEMENTS[name] in counter:
                condensced_results["num_predictions"].append({
                    name :  int(counter[SUB_ELEMENTS[name]])
                })
            else:
                condensced_results["num_predictions"].append({
                    name :  0
                })
            
        for k in range(len(r_b)):
            box = r_b[k]/224.
            local_result = {
                "bndbox" : {
                    "xmin": str(box[0]),
                    "ymin": str(box[1]),
                    "ymax": str(box[3]),
                    "xmax": str(box[2])
                },
                "score" : str(scores[k]),
                "class" : SUB_ELEMENTS_REVERSED[classes[k]]
            }
            condensced_results["object"].append(local_result)

        local_path = os.path.join(JSON_PATH, 'train/'+information_about_class[condensced_results["true_label"]] + img_name.split('/')[-1][:-4] + '.json')
        with open(local_path, 'w') as fp:
            json.dump(condensced_results, fp)