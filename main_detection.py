import os
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.config import *
from utils.dataloader import *
from utils.engine import train_one_epoch, evaluate

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
    train_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA, 'train.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)
    val_loader = ArchitectureDetectionDataset(os.path.join(PATH_DATA,'val.csv'), os.path.join(PATH_DATA, CSV_XML), transform_detection)

    optimizer = torch.optim.SGD(detector.parameters(), lr=0.003, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1, last_epoch=-1)

    best_acc = 0

    for i in range(1):
        print('Epoch {}/{}'.format(i+1, N_EPOCHS))
        print('-' * 10)
        #train_one_epoch(detector, optimizer, train_loader, "cuda", i, 180)
        evaluate(detector, val_loader, device="cuda")
        scheduler.step()
    
    torch.save(detector.state_dict(), DETECTOR_PATH)