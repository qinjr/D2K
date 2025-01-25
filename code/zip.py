import numpy as np
import pickle as pkl
import os
import random
import argparse
from tqdm import tqdm
from models import FM
import yaml
from utils.yaml_loader import get_yaml_loader
from dataloader import Dataloader
import torch

def get_model(model_name, model_config, data_config):
    model_name = model_name.lower()
    if model_name == 'fm':
        return FM(model_config, data_config)
    else:
        print('wrong model name: {}'.format(model_name))
        exit(1)

def random_select(in_files, out_file, random_ratio=0.1):
    x_user_list, x_item_list, x_context_list, y_list, hist_list, hist_len_list = [], [], [], [], [], []
    for in_file in in_files:
        print('begin random sampling on file: {}'.format(in_file))
        with open(in_file, 'rb') as f:
            x_user, x_item, x_context, y, hist, hist_len = pkl.load(f)
        size = x_user.shape[0]
        random_size = int(random_ratio * size)
        random_idx = np.arange(size)
        np.random.shuffle(random_idx)
        random_idx = random_idx[:random_size]

        x_user_list.append(x_user[random_idx])
        x_item_list.append(x_item[random_idx])
        x_context_list.append(x_context[random_idx])
        y_list.append(y[random_idx])
        hist_list.append(hist[random_idx])
        hist_len_list.append(hist_len[random_idx])

    x_user_sample = np.concatenate(x_user_list, axis=0)
    x_item_sample = np.concatenate(x_item_list, axis=0)
    x_context_sample = np.concatenate(x_context_list, axis=0)
    y_sample = np.concatenate(y_list, axis=0)
    hist_sample = np.concatenate(hist_list, axis=0)
    hist_len_sample = np.concatenate(hist_len_list, axis=0)

    print('shape after sampling:')
    print(x_user_sample.shape)

    with open(out_file, 'wb') as f:
        pkl.dump([x_user_sample, x_item_sample, x_context_sample, y_sample, hist_sample, hist_len_sample], f)
    print('random sampled file-{} has been dumpped'.format(out_file))


def svp_cf(dataloader, data_config, model_config, out_file, proxy_model_path, ratio=0.1, gpu_num=0):
    device = 'cuda:{}'.format(gpu_num)
    model = get_model(data_config['svp_cf_proxy_model_name'], model_config, data_config).to(device)
    # load the model from saved path
    checkpoint = torch.load(data_config['svp_cf_proxy_model_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('load the model from {}'.format(data_config['svp_cf_proxy_model_path']))

    iter_data = (
        tqdm(
            dataloader,
            total=len(dataloader),
            ncols=100,
            position=0, 
            leave=True
        )
    )

    errors = []
    x_user_list, x_item_list, x_context_list, y_list, hist_list, hist_len_list = [], [], [], [], [], []

    for batch_data in iter_data:
        x_user, x_item, x_context, y, user_hist, hist_len, klg = batch_data
        x_user_list.append(x_user)
        x_item_list.append(x_item)
        x_context_list.append(x_context)
        y_list.append(y)
        hist_list.append(user_hist)
        hist_len_list.append(hist_len)

        x_user = x_user.to(device)
        x_item = x_item.to(device)
        x_context = x_context.to(device)

        if model.use_hist:
            user_hist = user_hist.to(device)
            hist_len = hist_len.to(device)
        if model.use_klg:
            klg = klg.float().to(device)
            
        pred = model(x_user, x_item, x_context, user_hist, hist_len, klg).cpu().detach().numpy()
        errors.append(np.abs(y-pred))
    print('proxy scoring completed')

    errors = np.concatenate(errors, axis=0)
    y = np.concatenate(y_list, axis=0)

    index_pos = np.argsort(-errors[y==1]) # from large to small
    index_pos = index_pos[:int(index_pos.shape[0] * ratio)]
    print(index_pos.shape)
    index_neg = np.argsort(-errors[y==0]) # from large to small
    index_neg = index_neg[:int(index_neg.shape[0] * ratio)]
    print(index_neg.shape)
    index = np.concatenate([index_pos, index_neg], axis=0)

    x_user = np.concatenate(x_user_list, axis=0)[index]
    x_item = np.concatenate(x_item_list, axis=0)[index]
    x_context = np.concatenate(x_context_list, axis=0)[index]
    y = y[index]
    hist = np.concatenate(hist_list, axis=0)[index]
    hist_len = np.concatenate(hist_len_list, axis=0)[index]

    with open(out_file, 'wb') as f:
        pkl.dump([x_user, x_item, x_context, y, hist, hist_len], f)
    print('SVP-CF file-{} has been dumpped'.format(out_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ad')
    parser.add_argument('-m', '--mode', type=str, help='mode name', default='random')
    parser.add_argument('-in', '--in_blks', type=str, help='input blk numbers', default='0,1')
    args = parser.parse_args()

    in_blks = args.in_blks.split(',')

    root_path = '..'    
    os.chdir(root_path)
    DATA_DIR = 'data/{}/feateng_data'.format(args.dataset)
    if args.mode == 'random':
        in_files = [os.path.join(DATA_DIR, 'dataset_{}.pkl'.format(i)) for i in in_blks]
        out_file = os.path.join(DATA_DIR, 'dataset_random.pkl')
        random_select(in_files, out_file, random_ratio=0.1)
    elif args.mode == 'svp':
        # get configs
        loader = get_yaml_loader()
        data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
        with open(data_config_path, 'r') as f:
            data_config = yaml.load(f, Loader=loader)
        model_config_path = os.path.join('configs/model_configs', data_config['svp_cf_proxy_model_name'] + '.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=loader)[args.dataset]
        
        # get dataloader
        data_path = 'data/{}/feateng_data/dataset'.format(args.dataset)
        dataloader = Dataloader(data_path, in_blks, 10000, shuffle=False)

        # run svp cf
        out_file = os.path.join(DATA_DIR, 'dataset_svp.pkl')
        svp_cf(dataloader, data_config, model_config, out_file, data_config['svp_cf_proxy_model_path'], ratio=0.1, gpu_num=0)

