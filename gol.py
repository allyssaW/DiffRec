import os, logging, random
import numpy as np
import torch
from parse import parse_args
import multiprocessing


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def pLog(s: str):
    logging.info(s)

ARG = parse_args()

LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%m/%d %H:%M"
if ARG.log is not None:
    logging.basicConfig(filename=ARG.log, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
else:
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

SEED = ARG.seed
seed_torch(SEED)

EPOCH = ARG.epoch
BATCH_SZ = ARG.batch
TEST_BATCH_SZ = ARG.testbatch
DATA_PATH = '../data/'
FILE_PATH = './checkpoints/'
topk = ARG.topk
dataset = ARG.dataset
model = ARG.model

SAVE = ARG.save
LOAD = ARG.load
device = torch.device('cpu' if ARG.gpu is None else f'cuda:{ARG.gpu}')
os.makedirs(FILE_PATH, exist_ok=True)

all_dataset = ['gowalla', 'yelp2018', 'amazon-book']
all_models  = ['lgn']

conf = {'batch': ARG.batch, 'hidden': ARG.hidden, 'n_layers': ARG.layer,
        'dropout': ARG.dropout, 'keep_prob': ARG.keepprob, 'test_batch': ARG.testbatch,
        'multicore': ARG.multicore, 'lr': ARG.lr, 'decay': ARG.decay,
        'save': ARG.save, 'load': ARG.load}

GPU = torch.cuda.is_available()
device = torch.device(f'cuda:{ARG.gpu}' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2

if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model not in all_models:
    raise NotImplementedError(f"Haven't supported {model} yet!, try {all_models}")

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
