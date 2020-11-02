import cv2
import numpy as np 
import pandas as pd
from utils.parse_xml import parseXML
from utils.config import *
import cv2

fileToDraw = "/home/jules/Documents/Stage 4A/Data/Dataset-IGRB1092_14cls/03.renacentista/5bae6379a7219.jpg"
xmlToDraw = "/home/jules/Documents/Stage 4A/Data/Dataset-IGRB1092_14cls/03.renacentista/xml/5bae6379a7219.xml"

img = cv2.imread(fileToDraw)

_, _, _, info = parseXML(xmlToDraw, SUB_ELEMENTS)

for k in range(len(info['boxes'])):
    box = info['boxes'][k]
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 2)

cv2.imwrite('ourImageWithBox.jpg', img)

# cv2.imshow('imageWithBoxes', img)
# cv2.waitKey(0)