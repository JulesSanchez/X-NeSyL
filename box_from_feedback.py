import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, json

import torch
from torchvision import models, transforms

from skimage.segmentation import mark_boundaries

from utils.dataloader import val_transform
from utils.config import *



parser = argparse.ArgumentParser()
parser.add_argument("--interpreter", default="GradCAM")

args = parser.parse_args()

def batch_predict(images, model):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

#HERE LOAD IMAGE


idx2label = {
    0:'01.musulman',
    1:'02.gotico',
    2:'03.renacentista',
    3:'04.barroco'
}

model = models.resnet101(pretrained=True,num_classes=1000)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048,1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024,4)
)
model.load_state_dict(torch.load(MODEL_PATH))
for param in model.parameters():
    param.requires_grad = False
model.cuda()

if args.interpreter == 'GradCAM':
    from xdeep.xlocal.gradient.explainers import *
    explainer = ImageInterpreter(model)
    explanation = explainer.explain(image, method_name='gradcam', target_layer_name=None, viz=True, save_path=None) 

elif args.interpreter == 'GradCAMpp':
    from xdeep.xlocal.gradient.explainers import *
    explainer = ImageInterpreter(model)
    explanation = explainer.explain(image, method_name='gradcampp', target_layer_name=None, viz=True, save_path=None) 

elif args.interpreter == 'ScoreCAM':
    from xdeep.xlocal.gradient.explainers import *
    explainer = ImageInterpreter(model)
    explanation = explainer.explain(image, method_name='scorecam', target_layer_name=None, viz=True, save_path=None) 

elif args.interpreter == 'LIME':
    from lime import lime_image
    explainer = lime_image.LimeImageExplainer()
    #HERE PREP IMAGE
    explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=4, 
                                         hide_color=0, 
                                         num_samples=1000) # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)

elif args.interpreter == 'Mask':
    from utils.apply_mask import *
    tv_beta = 3
	learning_rate = 0.1
	max_iterations = 500
	l1_coeff = 0.01
	tv_coeff = 0.2
    original_img = cv2.imread(sys.argv[1], 1)
	original_img = cv2.resize(original_img, (224, 224))
	img = np.float32(original_img) / 255
	blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
	blurred_img2 = np.float32(cv2.medianBlur(original_img, 11))/255
	blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
	mask_init = np.ones((28, 28), dtype = np.float32)
	
	# Convert to torch variables
	img = preprocess_image(img)
	blurred_img = preprocess_image(blurred_img2)
	mask = numpy_to_torch(mask_init)

	if use_cuda:
		upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
	else:
		upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
	optimizer = torch.optim.Adam([mask], lr=learning_rate)

	target = torch.nn.Softmax()(model(img))
	category = np.argmax(target.cpu().data.numpy())
	print("Category with highest probability " + str(category))
	print("Optimizing.. ")

	for i in range(max_iterations):
		upsampled_mask = upsample(mask)
		# The single channel mask is used with an RGB image, 
		# so the mask is duplicated to have 3 channel,
		upsampled_mask = \
			upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
										upsampled_mask.size(3))
		
		# Use the mask to perturbated the input image.
		perturbated_input = img.mul(upsampled_mask) + \
							blurred_img.mul(1-upsampled_mask)
		
		noise = np.zeros((224, 224, 3), dtype = np.float32)
		cv2.randn(noise, 0, 0.2)
		noise = numpy_to_torch(noise)
		perturbated_input = perturbated_input + noise
		
		outputs = torch.nn.Softmax()(model(perturbated_input))
		loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
				tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Optional: clamping seems to give better results
		mask.data.clamp_(0, 1)

	upsampled_mask = upsample(mask)
