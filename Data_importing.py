import kagglehub
import shutil
import os

path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
shutil.move(path,os.getcwd())