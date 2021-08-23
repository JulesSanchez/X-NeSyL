import json, csv, os
from utils.knowledge_graph import knowledge_graph_monumenai
import numpy as np 

def compute_pred(results):
    contrib = [0,0,0,0]
    for i in range(len(knowledge_graph_monumenai)):
        for k in range(len(knowledge_graph_monumenai[i])):
            if knowledge_graph_monumenai[i,k]>0:
                contrib[k] += results[i]
    return np.argmax(contrib)

with open("result/01.musulmandic_res.json") as jsonFile:
    musulman = json.load(jsonFile)
    jsonFile.close()

with open("result/02.goticodic_res.json") as jsonFile:
    gothic = json.load(jsonFile)
    jsonFile.close()

with open("result/03.renacentistadic_res.json") as jsonFile:
    renaissance = json.load(jsonFile)
    jsonFile.close()

with open("result/04.barrocodic_res.json") as jsonFile:
    baroque = json.load(jsonFile)
    jsonFile.close()

#compute_pred([0.1,2,0,0,0,0,0,0,3,0,0,0,0.3,1])
with open('csvs/monumenai/test.csv') as csv_file:
    test_files = csv.reader(csv_file, delimiter=',')
    c = 0
    trues = []
    preds = []
    test_data = []
    for line in test_files:
        if c>0:
            results = [0 for i in range(14)]
            trues.append(int(line[1]))
            if line[1] == '0':
                res = musulman[os.path.basename(os.path.normpath(line[0]))]
            if line[1] == '1':
                res = gothic[os.path.basename(os.path.normpath(line[0]))]
            if line[1] == '2':
                res = renaissance[os.path.basename(os.path.normpath(line[0]))]
            if line[1] == '3':
                res = baroque[os.path.basename(os.path.normpath(line[0]))]
            for k in range(len(res['scores'])):
                results[res['labels'][k]-1] += res['scores'][k]
            preds.append(compute_pred(results))
            test_data.append(results)
        else:
            c+=1

with open('csvs/monumenai/train.csv') as csv_file:
    test_files = csv.reader(csv_file, delimiter=',')
    c = 0
    train_data = []
    train_label = []
    for line in test_files:
        if c>0:
            results = [0 for i in range(14)]
            train_label.append(int(line[1]))
            if line[1] == '0':
                res = musulman[os.path.basename(os.path.normpath(line[0]))]
            if line[1] == '1':
                res = gothic[os.path.basename(os.path.normpath(line[0]))]
            if line[1] == '2':
                res = renaissance[os.path.basename(os.path.normpath(line[0]))]
            if line[1] == '3':
                res = baroque[os.path.basename(os.path.normpath(line[0]))]
            for k in range(len(res['scores'])):
                results[res['labels'][k]-1] += res['scores'][k]
            train_data.append(results)
        else:
            c+=1

# import keras 
# classificator = keras.Sequential()
# classificator.add(keras.layers.Dense(units=11, activation='relu', input_shape=(14,)))
# classificator.add(keras.layers.Dense(units=4, activation='softmax'))
# classificator.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = classificator.fit(np.array(train_data), keras.utils.to_categorical(train_label, num_classes=4), batch_size=64, epochs=150, verbose=0)
# loss, accuracy = classificator.evaluate(np.array(test_data), keras.utils.to_categorical(trues, num_classes=4), verbose=1)


from sklearn.metrics import accuracy_score

print(accuracy_score(trues,preds))
#print(accuracy)