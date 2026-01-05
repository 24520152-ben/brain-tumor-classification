import os

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
IMG_SIZE = (224, 224)

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PATH_VGG = os.path.join(PROJECT_ROOT, 'models', 'VGG19_Best_Weights', 'VGG19_block_1_fold_1.keras')
PATH_MOBILENET = os.path.join(PROJECT_ROOT, 'models', 'MobileNet_Best_Weights', 'MobileNet_block_1_fold_1.keras')