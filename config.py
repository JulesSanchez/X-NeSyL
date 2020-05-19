PATH_DATA = '../Data/Dataset-IGRB1092_14cls'
FOLDERS_DATA = ['01.musulman','02.gotico','03.renacentista','04.barroco']
SUB_ELEMENTS = {
    'arco-herradura':0,
    'dintel-adovelado':1,
    'arco-lobulado':2,
    'arco-medio-punto':3,
    'arco-apuntado':4,
    'vano-adintelado':5,
    'fronton':6,
    'arco-conopial':7,
    'arco-trilobulado':8,
    'serliana':9,
    'ojo-de-buey':10,
    'fronton-curvo':11,
    'fronton-partido':12,
    'columna-salomonica':13
}

CSV_IMG = 'files_img.csv'
CSV_XML = 'files_xml.csv'

N_EPOCHS = 100
BATCH_SIZE = 8
MODEL_PATH = './model/model_resnet101.pth'