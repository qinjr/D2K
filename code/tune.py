import argparse
import os
import yaml
import numpy as np
from recbole.utils import set_color
# from torch.utils.data import DataLoader
from recbole.utils import init_seed
from dataloader import Dataloader, DataloaderKLG
from utils.log import init_logger
from utils.yaml_loader import get_yaml_loader
from logging import getLogger
from models import *
from trainer import Trainer
import pickle as pkl

def get_model(model_name, model_config, data_config, klg_type):
    model_name = model_name.lower()
    if model_name == 'only_klg_lr':
        return ONLY_KLG_LR(model_config, data_config, klg_type)
    elif model_name == 'lr':
        return LR(model_config, data_config)
    elif model_name == 'lr_klg':
        return LR_KLG(model_config, data_config, klg_type)
    elif model_name == 'deepfm':
        return DeepFM(model_config, data_config)
    elif model_name == 'deepfm_klg':
        return DeepFM_KLG(model_config, data_config, klg_type)
    elif model_name == 'din':
        return DIN(model_config, data_config)
    elif model_name == 'din_klg':
        return DIN_KLG(model_config, data_config, klg_type)
    elif model_name == 'addencoder':
        return AddEncoder(model_config, data_config)
    elif model_name == 'addencoderwithhist':
        return AddEncoderWithHist(model_config, data_config)
    elif model_name == 'transformer':
        return Transformer(model_config, data_config)
    else:
        print('wrong model name: {}'.format(model_name))
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='model name', default='deepfm')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ad')
    parser.add_argument('-trs', '--train_split', type=str, help='the training data splits')
    parser.add_argument('-tes', '--test_split', type=str, help='the test data splits')

    args = parser.parse_args()

    train_split = args.train_split.split(',')
    test_split = args.test_split.split(',')
    
    # go to root path
    root_path = '..'
    os.chdir(root_path)
    data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    train_config_path = os.path.join('configs/train_configs', args.dataset + '.yaml')
    model_config_path = os.path.join('configs/model_configs', args.model + '.yaml')

    loader = get_yaml_loader()
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=loader)
    with open(train_config_path, 'r') as f:
        train_config = yaml.load(f, Loader=loader)
    with open(model_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=loader)[args.dataset]

    run_config = {'model': args.model,
                  'dataset': args.dataset}
    init_seed(train_config['seed'], train_config['reproducibility'])
    # logger initialization
    init_logger(run_config)
    logger = getLogger()
    logger.info('train_split is {}'.format(args.train_split))

    logger.info(run_config)
    logger.info(train_config)

    # dataloaders: train and test
    data_path = 'data/{}/feateng_data/dataset'.format(args.dataset)
    if train_config['klg_type'] == 'pos_ratio':
        klg_path = 'data/{}/feateng_data/klg_pos_ratio'.format(args.dataset)
    elif train_config['klg_type'] == 'inner_product':
        klg_path = 'data/{}/feateng_data/klg_inner_product'.format(args.dataset)
    elif train_config['klg_type'] == 'encoder_vec':
        klg_path = 'data/{}/feateng_data/klg_encoder_vec'.format(args.dataset)
    else:
        klg_path = None
        print('not supported {} as a klg type'.format(train_config['klg_type']))

    if args.dataset == 'ml-1m':
        batch_sizes = [256, 512, 1024] # ML-1M
        lrs = [1e-3, 5e-4, 1e-4] # ML-1M
    elif args.dataset == 'ad':
        batch_sizes = [512, 1024] # Ad
        lrs = [1e-3, 1e-4] # Ad
    elif args.dataset == 'eleme':
        batch_sizes = [1024, 2048] # Eleme
        lrs = [5e-4, 1e-4] # Eleme
        # # for AddEncoder & AddEncoder2
        # batch_sizes = [1024] # Eleme
        # lrs = [5e-6, 1e-6, 5e-7] # Eleme
    elif args.dataset == 'tenrec':
        # batch_sizes = [1024, 2048] # Tenrec
        # lrs = [1e-3, 1e-4] # Tenrec
        batch_sizes = [2048, 4096] #see deepfm on dataset_1 
        lrs = [1e-4, 1e-5] # Tenrec

    # dataloaders: train and test
    if 'klg' in args.model:
        train_dl = DataloaderKLG(data_path, klg_path, train_split, batch_sizes[0], shuffle=True)
        test_dl = DataloaderKLG(data_path, klg_path, test_split, train_config['eval_batch_size'], shuffle=False)
    else:
        train_dl = Dataloader(data_path, train_split, batch_sizes[0], shuffle=True)
        test_dl = Dataloader(data_path, test_split, train_config['eval_batch_size'], shuffle=False)

    for batch_size in batch_sizes:
        for lr in lrs:
            train_config['train_batch_size'] = batch_size
            train_config['learning_rate'] = lr

            train_dl.update_batch_size(batch_size)
            
            logger.info('exp with lr={}, batch_size={}'.format(lr, batch_size))
            # get model
            model = get_model(run_config['model'], model_config, data_config, train_config['klg_type']).to(train_config['device'])
            logger.info(model)

            # get trainer and fit
            trainer = Trainer(train_config, model, args.dataset)
            best_eval_result = trainer.fit(train_dl, test_dl, saved=False)
