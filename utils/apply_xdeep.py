from xdeep.xlocal.gradient.explainers import *
import torchvision.models as models
from .config import *
import torch 

img_path = PATH_DATA+'/01.musulman/5b8ecf08e1e7f.jpg'
image = load_image(img_path)
model = models.resnet101(pretrained=True,num_classes=1000)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048,512),
    torch.nn.Linear(512,256),
    torch.nn.Linear(256,4)
)
model.load_state_dict(torch.load(MODEL_PATH))
# build the xdeep explainer
model_explainer = ImageInterpreter(model)

# generate the local interpretation
model_explainer.explain(image, method_name='gradcampp
', target_layer_name=None, viz=True, save_path='here.jpg') 