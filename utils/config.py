PATH_DATA = '../Data/Dataset-IGRB1092_14cls'
FOLDERS_DATA = ['01.musulman','02.gotico','03.renacentista','04.barroco']
SUB_ELEMENTS = {
    'arco-herradura':1,
    'dintel-adovelado':2,
    'arco-lobulado':3,
    'arco-medio-punto':4,
    'arco-apuntado':5,
    'vano-adintelado':6,
    'fronton':7,
    'arco-conopial':8,
    'arco-trilobulado':9,
    'serliana':10,
    'ojo-de-buey':11,
    'fronton-curvo':12,
    'fronton-partido':13,
    'columna-salomonica':14
}

SUB_ELEMENTS_REVERSED = {
    1:'arco-herradura',
    2:'dintel-adovelado',
    3:'arco-lobulado',
    4:'arco-medio-punto',
    5:'arco-apuntado',
    6:'vano-adintelado',
    7:'fronton',
    8:'arco-conopial',
    9:'arco-trilobulado',
    10:'serliana',
    11:'ojo-de-buey',
    12:'fronton-curvo',
    13:'fronton-partido',
    14:'columna-salomonica'
}

ELEMENTS_LABEL = {
    1:['01.musulman'],
    2:['01.musulman'],
    3:['01.musulman'],
    4: ['03.renacentista','04.barroco'],
    5:['02.gotico'],
    6:['03.renacentista','04.barroco'],
    7: ['03.renacentista'], 
    8:['02.gotico'],
    9:['02.gotico'],
    10:['03.renacentista'],
    11: ['03.renacentista','04.barroco'],
    12:['03.renacentista'],
    13:['04.barroco'],
    14:['04.barroco']
}

CSV_IMG = 'files_img.csv'
CSV_XML = 'files_xml.csv'

N_EPOCHS = 20
BATCH_SIZE = 8
MODEL_PATH = './model/model_resnet101.pth'
DETECTOR_PATH = './model/model_fasterRCNN_bbox_shap.pth'
JSON_PATH = '../Results/json_detection'
TMP_PATH = './tmp'

PASCAL_EL_DIC = {'Bird': 0, 'Aeroplane': 1, 'Cat': 2, 'Dog': 3, 'Sheep': 4, 'Train': 5, 'Bicycle': 6, 'Horse': 7, 'Bottle': 8, 'Person': 9, 'Car': 10, 'diningtable': 11, 'Pottedplant': 12, 'Motorbike': 13, 'Sofa': 14, 'Boat': 15, 'Cow': 16, 'Chair': 17, 'Bus': 18, 'Tvmonitor': 19}
PASCAL_PART_DIC = {'Arm': 1, 'Engine': 2, 'Coach': 3, 'Tail': 4, 'Pot': 5, 'Cap': 6, 'Ear': 7, 'Horn': 8, 'Ebrow': 9, 'Nose': 10, 'Torso': 11, 'Head': 12, 'Body': 13, 'Muzzle': 14, 'Beak': 15, 'Hand': 16, 'Hair': 17, 'Neck': 18, 'Foot': 19, 'Stern': 20, 'Artifact_Wing': 21, 'Locomotive': 22, 'License_plate': 23, 'Screen': 24, 'Mirror': 25, 'Saddle': 26, 'Hoof': 27, 'Door': 28, 'Leg': 29, 'Plant': 30, 'Mouth': 31, 'Animal_Wing': 32, 'Eye': 33, 'Chain_Wheel': 34, 'Bodywork': 35, 'Handlebar': 36, 'Headlight': 37, 'Wheel': 38, 'Window': 39, 'diningtable': 40, 'Sofa':41, 'Boat':42, 'Chair':43, 'Tvmonitor':44}
PASCAL_PART_DIC_REVERSED = {1: 'Arm', 2: 'Engine', 3: 'Coach', 4: 'Tail', 5: 'Pot', 6: 'Cap', 7: 'Ear', 8: 'Horn', 9: 'Ebrow', 10: 'Nose', 11: 'Torso', 12: 'Head', 13: 'Body', 14: 'Muzzle', 15: 'Beak', 16: 'Hand', 17: 'Hair', 18: 'Neck', 19: 'Foot', 20: 'Stern', 21: 'Artifact_Wing', 22: 'Locomotive', 23: 'License_plate', 24: 'Screen', 25: 'Mirror', 26: 'Saddle', 27: 'Hoof', 28: 'Door', 29: 'Leg', 30: 'Plant', 31: 'Mouth', 32: 'Animal_Wing', 33: 'Eye', 34: 'Chain_Wheel', 35: 'Bodywork', 36: 'Handlebar', 37: 'Headlight', 38: 'Wheel', 39: 'Window', 40:'diningtable', 41:'Sofa', 42:'Boat', 43:'Chair', 44:'Tvmonitor'}
PASCAL_KNOWLEDGE_BASE = {'Bird': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Beak', 'Animal_Wing', 'Head'],
 'Aeroplane': ['Stern', 'Engine', 'Wheel', 'Artifact_Wing', 'Body'],
 'Cat': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
 'Dog': ['Torso', 'Muzzle', 'Nose', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
 'Sheep': ['Torso', 'Tail', 'Muzzle', 'Neck', 'Eye', 'Horn', 'Leg', 'Ear', 'Head'],
 'Train': ['Locomotive', 'Coach', 'Headlight'],
 'Bicycle': ['Chain_Wheel', 'Saddle', 'Wheel', 'Handlebar'],
 'Horse': ['Hoof', 'Torso', 'Muzzle', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'], 
 'Bottle': ['Cap', 'Body'], 
 'Person': ['Ebrow', 'Foot', 'Arm', 'Torso', 'Nose', 'Hair', 'Hand', 'Neck', 'Eye', 'Leg', 'Ear', 'Head', 'Mouth'], 
 'Car': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'], 
 'diningtable': ['diningtable'], 
 'Pottedplant': ['Pot', 'Plant'], 
 'Motorbike': ['Wheel', 'Headlight', 'Saddle', 'Handlebar'], 
 'Sofa': ['Sofa'], 
 'Boat': ['Boat'], 
 'Cow': ['Torso', 'Muzzle', 'Tail', 'Horn', 'Eye', 'Neck', 'Leg', 'Ear', 'Head'], 
 'Chair': ['Chair'], 
 'Bus': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'], 
 'Tvmonitor': ['Screen', 'Tvmonitor']}

PATH_PASCAL = '/home/jules/Documents/Stage 4A/Data/pascalPartDataset/'
PASCAL_XML = 'AnnotationsPascalPart'
PASCAL_IMG = 'ImagesPascalPart'
PASCAL_LIST = 'singular_pascal.txt'
train_pascal = PATH_PASCAL + 'train.txt'
val_pascal = PATH_PASCAL + 'val.txt'
test_pascal = PATH_PASCAL + 'test.txt'
