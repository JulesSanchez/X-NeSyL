import csv 
from sklearn.metrics import confusion_matrix
import numpy as np 

STYLES_HOTONE_ENCODE = {'M' : 0, 'G' : 1, 'R' : 2, 'B' : 3}
names = []
truth = []
pred = []

with open('../MonuMAI-AutomaticStyleClassification/good_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        names.append(row[0][1:])
        truth.append(STYLES_HOTONE_ENCODE[row[0][0]])
        pred.append([float(row[1]),float(row[2]),float(row[3]),float(row[4])])


with open('../MonuMAI-AutomaticStyleClassification/bad_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        names.append(row[0][1:])
        truth.append(STYLES_HOTONE_ENCODE[row[0][0]])
        pred.append([float(row[1]),float(row[2]),float(row[3]),float(row[4])])

truth = np.array(truth)
pred = np.array(pred)
pred_local = np.argmax(pred,axis=1)

cm = confusion_matrix(truth,pred_local)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=STYLES_HOTONE_ENCODE.keys())
disp.plot(include_values=True, cmap='viridis')
plt.show()