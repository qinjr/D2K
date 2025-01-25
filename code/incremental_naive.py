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
    elif model_name == 'ipnn':
        return IPNN(model_config, data_config)
    elif model_name == 'uiipnn':
        return UIIPNN(model_config, data_config)
    elif model_name == 'addencoder':
        return AddEncoder(model_config, data_config)
    elif model_name == 'addencoder2':
        return AddEncoder2(model_config, data_config)
    elif model_name == 'dcnv2':
        return DCNV2(model_config, data_config)
    else:
        print('wrong model name: {}'.format(model_name))
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='model name', default='deepfm')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ad')
    parser.add_argument('-fixed_trs', '--fixed_train_split', type=str, help='the fixed training data splits')
    parser.add_argument('-incre_trs', '--incre_train_split', type=str, help='the incremental training data splits')
    parser.add_argument('-tes', '--test_split', type=str, help='the test data splits')
    parser.add_argument('-it', '--num_iter', type=int, help='times of training', default=5)

    args = parser.parse_args()

    fixed_train_split = list(map(int, args.fixed_train_split.split(',')))
    incre_train_split = list(map(int, args.incre_train_split.split(',')))
    test_split = list(map(int, args.test_split.split(',')))
    all_incre_split = incre_train_split + test_split

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
    logger.info('fixed_train_split is {}'.format(args.fixed_train_split))
    logger.info('incre_train_split is {}'.format(args.incre_train_split))

    logger.info(run_config)
    logger.info(train_config)

    # dataloaders: train and test
    data_path = 'data/{}/feateng_data/dataset'.format(args.dataset)
    
    aucs = []
    lls = []
    for _ in range(args.num_iter):
        # get model
        model = get_model(run_config['model'], model_config, data_config, train_config['klg_type']).to(train_config['device'])
        logger.info(model)
        trainer = Trainer(train_config, model, args.dataset)

        # fixed training part
        train_dl = Dataloader(data_path, fixed_train_split, train_config['train_batch_size'], shuffle=True)
        test_dl = Dataloader(data_path, [incre_train_split[0]], train_config['eval_batch_size'], shuffle=False)
        _ = trainer.fit(train_dl, test_dl)

        # incremental training part
        # incrementally getting the train and test datasets
        for i in range(len(all_incre_split) - 1):
            train_dl = Dataloader(data_path, [all_incre_split[i]], train_config['train_batch_size'], shuffle=True)
            test_dl = Dataloader(data_path, [all_incre_split[i + 1]], train_config['eval_batch_size'], shuffle=False)

            trainer.self_resume_checkpoint()
            # get trainer and fit
            best_eval_result = trainer.fit(train_dl, test_dl)
        
        # last test auc and ll are the final results
        aucs += [best_eval_result['AUC']]
        lls += [best_eval_result['LL']]
    
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)
    ll_mean = np.mean(lls)
    ll_std = np.std(lls)

    final_output = set_color('Final results', 'blue') + ': \n' + 'AUC-MEAN: {}, AUC-STD: {}, LL-MEAN: {}, LL-STD: {}'.format(auc_mean, auc_std, ll_mean, ll_std)
    logger.info(final_output)
