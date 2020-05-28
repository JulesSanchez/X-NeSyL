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

ELEMENTS_LABEL = {
    'arco-herradura':['01.musulman'],
    'dintel-adovelado':['03.renacentista','04.barroco'],
    'arco-lobulado':['01.musulman'],
    'arco-medio-punto': ['03.renacentista','04.barroco'],
    'arco-apuntado':['02.gotico'],
    'vano-adintelado':['01.musulman'],
    'fronton': ['03.renacentista'], 
    'arco-conopial':['02.gotico'],
    'arco-trilobulado':['02.gotico'],
    'serliana':['03.renacentista'],
    'ojo-de-buey': ['03.renacentista','04.barroco'],
    'fronton-curvo':'03.renacentista',
    'fronton-partido':['04.barroco]',
    'columna-salomonica':['04.barroco']
}

CSV_IMG = 'files_img.csv'
CSV_XML = 'files_xml.csv'

N_EPOCHS = 25
BATCH_SIZE = 8
MODEL_PATH = './model/model_resnet101.pth'