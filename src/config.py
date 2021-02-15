import os
import sys
import logging
import numpy as np
import tensorflow as tf

FIDL_ADD_IFS_BRANCH = False
FIDL_RS = np.random.RandomState(0)
FIDL_PROJECT_DIR = "/home/maks/Scripts/Search_Net/Search_Net_PFS"
FIDL_TENSORBOARD_LOG_DIR = os.path.join(FIDL_PROJECT_DIR,"tb_logs")
FIDL_LOG_DIR = os.path.join(FIDL_PROJECT_DIR,"logs")
FIDL_MODEL_DIR = os.path.join(FIDL_PROJECT_DIR,"saved_models")

FIDL_LOG_TO_FILE = False

if FIDL_LOG_TO_FILE == True:
    import datetime
    original_stdout = sys.stdout # Save a reference to the original standard output
    filename = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")+".log"
    filename = os.path.join(FIDL_LOG_DIR,"logs",filename)
    logging.basicConfig(filename=filename,  level=logging.DEBUG,format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

else:
    logging.basicConfig(level=logging.DEBUG,format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

logging.info("Using tf version: "+str(tf.__version__))