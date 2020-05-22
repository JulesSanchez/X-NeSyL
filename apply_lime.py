
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

from dataloader import val_transform

from lime import lime_image

from skimage.segmentation import mark_boundaries

from config import *

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

def get_input_tensors(img):
    # unsqeeze converts single image to batch of 1
    return val_transform(img).unsqueeze(0).cuda()

idx2label = {
    0:'01.musulman',
    1:'02.gotico',
    2:'03.renacentista',
    3:'04.barroco'
}

img = get_image(PATH_DATA+'/01.musulman/5b8ecf08e1e7f.jpg')
img_t = get_input_tensors(img)


model = models.resnet101(pretrained=True,num_classes=1000)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048,512),
    torch.nn.Linear(512,256),
    torch.nn.Linear(256,4)
)
model.load_state_dict(torch.load(MODEL_PATH))

model.cuda()

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=4, 
                                         hide_color=0, 
                                         num_samples=1000) # number of images that will be sent to classification function

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255, mask)
plt.imshow(img_boundry1)
plt.show()