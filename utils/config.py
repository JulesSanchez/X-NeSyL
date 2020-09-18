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

N_EPOCHS = 1
BATCH_SIZE = 8
MODEL_PATH = './model/model_resnet101.pth'
DETECTOR_PATH = './model/model_fasterRCNN_bbox_shap.pth'
JSON_PATH = '../Results/json_detection'
TMP_PATH = './tmp'

PASCAL_EL_DIC = {'diningtable': 0, 'Bodywork': 1, 'Aeroplane': 2, 'Boat': 3, 'Hand': 4, 'Body': 5, 'Bus': 6, 'Arm': 7, 'Coach': 8, 'Tvmonitor': 9, 'Neck': 10, 'Stern': 11, 'Screen': 12, 'Leg': 13, 'Wheel': 14, 'Horn': 15, 'Foot': 16, 'Saddle': 17, 'Bottle': 18, 'License_plate': 19, 'Beak': 20, 'Locomotive': 21, 'Mouth': 22, 'Person': 23, 'Motorbike': 24, 'Ear': 25, 'Nose': 26, 'Pot': 27, 'Engine': 28, 'Window': 29, 'Mirror': 30, 'Pottedplant': 31, 'Muzzle': 32, 'Dog': 33, 'Torso': 34, 'Cat': 35, 'Car': 36, 'Train': 37, 'Handlebar': 38, 'Bicycle': 39, 'Animal_Wing': 40, 'Door': 41, 'Cap': 42, 'Sheep': 43, 'Eye': 44, 'Headlight': 45, 'Chair': 46, 'Artifact_Wing': 47, 'Cow': 48, 'Plant': 49, 'Chain_Wheel': 50, 'Ebrow': 51, 'Hoof': 52, 'Horse': 53, 'Tail': 54, 'Bird': 55, 'Hair': 56, 'Sofa': 57, 'Head': 58}
