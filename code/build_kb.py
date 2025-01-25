import pickle as pkl
import numpy as np
import argparse
import os
import yaml
import torch
from tqdm import tqdm
from utils.yaml_loader import get_yaml_loader
from dataloader import KBEntryloader, Dataloader
from models import *

def get_model(model_name, model_config, data_config):
    model_name = model_name.lower()
    if model_name == 'deepfm':
        return DeepFM(model_config, data_config)
    elif model_name == 'din':
        return DIN(model_config, data_config)
    elif model_name == 'addencoder':
        return AddEncoder(model_config, data_config)
    elif model_name == 'addencoderwithhist':
        return AddEncoderWithHist(model_config, data_config)
    elif model_name == 'transformer':
        return Transformer(model_config, data_config)
    elif model_name == 'gru4rec':
        return GRU4Rec(model_config, data_config)
    else:
        print('wrong model name: {}'.format(model_name))
        exit(1)


def build_pos_ratio_as_kb_wohist(dataset_files, kb_file, data_config):
    kb = {}
    for dataset_f in dataset_files:
        with open(dataset_f, 'rb') as f:
            print('start adding entity for dataset file: {}'.format(dataset_f))
            data = pkl.load(f)
            x_user, x_item, x_context, y = data[0], data[1], data[2], data[3]
            
            for i in data_config['klg_user_feats']:
                for j in data_config['klg_item_feats']:
                    for k in data_config['klg_context_feats']:
                        print('begin inserting user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                        user_feat_i = x_user[:,i].tolist()
                        item_feat_j = x_item[:,j].tolist()
                        context_feat_k = x_context[:,k].tolist()
                        for p, tup in tqdm(enumerate(zip(user_feat_i, item_feat_j, context_feat_k))):
                            label = y[p]
                            if label == 1:
                                if tup not in kb:
                                    kb[tup] = [1,0] # [pos_freq, neg_freq]
                                else:
                                    kb[tup][0] += 1
                            else:
                                if tup not in kb:
                                    kb[tup] = [0,1]
                                else:
                                    kb[tup][1] += 1

    print('the number of entries in the knowledge base: {}'.format(len(kb)))

    # transform to frequency
    for k in kb:
        kb[k] = kb[k][0] / sum(kb[k])
    with open(kb_file, 'wb') as f:
        pkl.dump(kb, f)

def build_pos_ratio_as_kb(dataset_files, kb_file, data_config):
    padding_id = data_config['padding_id']
    kb = {}
    for dataset_f in dataset_files:
        with open(dataset_f, 'rb') as f:
            print('start adding entity for dataset file: {}'.format(dataset_f))
            x_user, x_item, x_context, y, hist, hist_len = pkl.load(f)
            # single value features
            
            for i in data_config['klg_user_feats']:
                for j in data_config['klg_item_feats']:
                    for k in data_config['klg_context_feats']:
                        print('begin inserting user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                        user_feat_i = x_user[:,i].tolist()
                        item_feat_j = x_item[:,j].tolist()
                        context_feat_k = x_context[:,k].tolist()
                        for p, tup in tqdm(enumerate(zip(user_feat_i, item_feat_j, context_feat_k))):
                            label = y[p]
                            if label == 1:
                                if tup not in kb:
                                    kb[tup] = [1,0] # [pos_freq, neg_freq]
                                else:
                                    kb[tup][0] += 1
                            else:
                                if tup not in kb:
                                    kb[tup] = [0,1]
                                else:
                                    kb[tup][1] += 1

            # multi value features
            for i in data_config['klg_item_feats']:
                for j in data_config['klg_hist_feats']:
                    for k in data_config['klg_context_feats']:
                        print('begin inserting item feat {} + hist feat {} + context feat {} in the KB'.format(i, j, k))
                        item_feat_i = x_item[:,i].tolist()
                        user_seq_feat_j = hist[:,:,j].tolist()
                        context_feat_k = x_context[:,k].tolist()
                        for p, (item_feat_i_p, context_feat_k_p) in tqdm(enumerate(zip(item_feat_i, context_feat_k))):
                            label = y[p]
                            user_seq_feat_j_p = user_seq_feat_j[p]
                            for q in user_seq_feat_j_p:
                                if q == padding_id:
                                    continue
                                else:
                                    tup = (item_feat_i_p, q, context_feat_k_p)
                                    if label == 1:
                                        if tup not in kb:
                                            kb[tup] = [1,0]
                                        else:
                                            kb[tup][0] += 1
                                    else:
                                        if tup not in kb:
                                            kb[tup] = [0,1]
                                        else:
                                            kb[tup][1] += 1
    print('the number of entries in the knowledge base: {}'.format(len(kb)))

    # transform to frequency
    for k in kb:
        kb[k] = kb[k][0] / sum(kb[k])
    with open(kb_file, 'wb') as f:
        pkl.dump(kb, f)

def pos_ratio_kb_stat(kb_file):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
    weights = []
    for key in kb:
        weights.append(kb[key])
    print('avg weights is: {}'.format(sum(weights)/len(weights)))
    print('max weights is: {}'.format(max(weights)))
    print('min weights is: {}'.format(min(weights)))

def gen_kb_entry_wohist(dataset_files, kb_entry_file, data_config):
    kb_entry = {}
    for dataset_f in dataset_files:
        with open(dataset_f, 'rb') as f:
            print('start adding entity for dataset file: {}'.format(dataset_f))
            data = pkl.load(f)
            x_user, x_item, x_context, y = data[0], data[1], data[2], data[3]
            for i in data_config['klg_user_feats']:
                for j in data_config['klg_item_feats']:
                    for k in data_config['klg_context_feats']:
                        print('begin inserting user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                        user_feat_i = x_user[:,i].tolist()
                        item_feat_j = x_item[:,j].tolist()
                        context_feat_k = x_context[:,k].tolist()
                        for p, tup in tqdm(enumerate(zip(user_feat_i, item_feat_j, context_feat_k))):
                            kb_entry[tup] = 1
    print('the number of entries in the knowledge base: {}'.format(len(kb_entry)))
    kb_entry_mtx = np.array([ent for ent in kb_entry])
    print('shape of kb_entry matrix is:')
    print(kb_entry_mtx.shape)
    
    with open(kb_entry_file, 'wb') as f:
        pkl.dump(kb_entry_mtx, f)

def gen_kb_entry(dataset_files, kb_entry_file, data_config):
    padding_id = data_config['padding_id']
    kb_entry = {}
    for dataset_f in dataset_files:
        with open(dataset_f, 'rb') as f:
            print('start adding entity for dataset file: {}'.format(dataset_f))
            x_user, x_item, x_context, y, hist, _ = pkl.load(f)
            for i in data_config['klg_user_feats']:
                for j in data_config['klg_item_feats']:
                    for k in data_config['klg_context_feats']:
                        print('begin inserting user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                        user_feat_i = x_user[:,i].tolist()
                        item_feat_j = x_item[:,j].tolist()
                        context_feat_k = x_context[:,k].tolist()
                        for p, tup in tqdm(enumerate(zip(user_feat_i, item_feat_j, context_feat_k))):
                            kb_entry[tup] = 1
            for i in data_config['klg_item_feats']:
                for j in data_config['klg_hist_feats']:
                    for k in data_config['klg_context_feats']:
                        print('begin inserting item feat {} + hist feat {} + context feat {} in the KB'.format(i, j, k))
                        item_feat_i = x_item[:,i].tolist()
                        user_seq_feat_j = hist[:,:,j].tolist()
                        context_feat_k = x_context[:,k].tolist()
                        for p, (item_feat_i_p, context_feat_k_p) in tqdm(enumerate(zip(item_feat_i, context_feat_k))):
                            user_seq_feat_j_p = user_seq_feat_j[p]
                            for q in user_seq_feat_j_p:
                                if q != padding_id:
                                    tup = (item_feat_i_p, q, context_feat_k_p)
                                    kb_entry[tup] = 1


    print('the number of entries in the knowledge base: {}'.format(len(kb_entry)))
    kb_entry_mtx = np.array([ent for ent in kb_entry])
    print('shape of kb_entry matrix is:')
    print(kb_entry_mtx.shape)
    
    with open(kb_entry_file, 'wb') as f:
        pkl.dump(kb_entry_mtx, f)

def build_inner_product_as_kb(model_config, data_config, kb_entry_file, 
                              kb_file, batch_size=2000, gpu_num=0):
    device = 'cuda:{}'.format(gpu_num)
    model = get_model(data_config['kb_inner_product_model_name'], model_config, data_config).to(device)
    # load the model from saved path
    checkpoint = torch.load(data_config['kb_inner_product_model_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('load the model from {}'.format(data_config['kb_inner_product_model_path']))
    
    kb = {}
    dl = KBEntryloader(batch_size, kb_entry_file)
    keys = []
    values = []
    for entry_batch in tqdm(dl):
        entry_batch_device = torch.from_numpy(entry_batch).to(device)
        embed_x = model.get_feat_embedding(entry_batch_device)
        inner_products = torch.sum(embed_x[:,0,:] * embed_x[:,1,:], dim=1)
        inner_products = inner_products.cpu().detach().numpy()#.tolist()

        # adding to kb
        # entry_batch = entry_batch.tolist()
        # for ent, inner in zip(entry_batch, inner_products):
        #     kb[tuple(ent)] = inner
        keys.append(entry_batch)
        values.append(inner_products)

    for key_batch, value_batch in tqdm(zip(keys, values), total=len(keys)):
        for key, value in zip(key_batch, value_batch):
            kb[tuple(key)] = float(value)

    print('number of kb enties is: {}'.format(len(kb)))
    
    with open(kb_file, 'wb') as f:
        pkl.dump(kb, f)

def build_encoder_vec_as_kb(model_config, data_config, kb_entry_file, 
                            kb_file, batch_size=2000, gpu_num=0):
    device = 'cuda:{}'.format(gpu_num)
    model = get_model(data_config['kb_encoder_model_name'], model_config, data_config).to(device)
    # load the model from saved path
    checkpoint = torch.load(data_config['kb_encoder_model_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('load the model from {}'.format(data_config['kb_encoder_model_path']))
    
    kb = {}
    dl = KBEntryloader(batch_size, kb_entry_file)
    for entry_batch in tqdm(dl):
        entry_batch_device = torch.from_numpy(entry_batch).to(device)
        klg_vecs = model.get_klg_vec(entry_batch_device)
        klg_vecs = klg_vecs.cpu().detach().numpy().tolist()

        # adding to kb
        entry_batch = entry_batch.tolist()
        for ent, klg_vec in zip(entry_batch, klg_vecs):
            kb[tuple(ent)] = klg_vec
    print('number of kb enties is: {}'.format(len(kb)))

    with open(kb_file, 'wb') as f:
        pkl.dump(kb, f)

def build_user_seq_rep_as_kb(model_config, data_config, data_path, split_idx,
                            kb_file, batch_size=2000, gpu_num=0):
    device = 'cuda:{}'.format(gpu_num)
    model = get_model(data_config['user_seq_encoder_model_name'], model_config, data_config).to(device)
    # load the model from saved path
    checkpoint = torch.load(data_config['user_seq_encoder_model_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('load the model from {}'.format(data_config['user_seq_encoder_model_path']))
    
    kb = {}
    dl = Dataloader(data_path, split_idx, batch_size, shuffle=False)
    for data in tqdm(dl):
        x_user, x_item, x_context, y, user_hist, hist_len, klg = data
        user_hist = user_hist.to(device)
        hist_len = hist_len.to(device)

        klg_vecs = model.get_user_seq_rep(user_hist, hist_len)
        klg_vecs = klg_vecs.cpu().detach().numpy().tolist()

        # adding to kb
        uids = x_user[:,0].tolist()
        for uid, klg_vec in zip(uids, klg_vecs):
            kb[uid] = klg_vec
    print('number of kb enties is: {}'.format(len(kb)))

    with open(kb_file, 'wb') as f:
        pkl.dump(kb, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ad')
    parser.add_argument('-m', '--kb_mode', type=str, help='kb mode: (pos_ratio, inner_product, encoder_vec)', default='pos_ratio')
    parser.add_argument('-kb', '--kb_files', type=str, help='dataset files to built the kb', default='0,1,2,3')
    args = parser.parse_args()
    
    kb_blk_list = list(map(int, args.kb_files.split(',')))

    root_path = '..'
    os.chdir(root_path)
    data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    loader = get_yaml_loader()
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=loader)

    dataset_file_prefix = 'data/{}/feateng_data/dataset_'.format(args.dataset)
    dataset_files = [dataset_file_prefix + '{}.pkl'.format(i) for i in kb_blk_list]
    print(dataset_files)

    if not os.path.exists('data/{}/feateng_data/kb_entry.pkl'.format(args.dataset)):
        gen_kb_entry(dataset_files, 'data/{}/feateng_data/kb_entry.pkl'.format(args.dataset), data_config)
    # if not os.path.exists('data/{}/feateng_data/kb_entry_wo.pkl'.format(args.dataset)):
    #     gen_kb_entry_wohist(dataset_files, 'data/{}/feateng_data/kb_entry_wohist.pkl'.format(args.dataset), data_config)

    if args.kb_mode == 'pos_ratio':
        kb_file = 'data/{}/feateng_data/kb_pos_ratio.pkl'.format(args.dataset)
        build_pos_ratio_as_kb(dataset_files, kb_file, data_config)
        # kb_file = 'data/{}/feateng_data/kb_pos_ratio_wohist.pkl'.format(args.dataset)
        # build_pos_ratio_as_kb_wohist(dataset_files, kb_file, data_config)
        pos_ratio_kb_stat(kb_file)
    
    elif args.kb_mode == 'inner_product':
        model_config_path = os.path.join('configs/model_configs', data_config['kb_inner_product_model_name'] + '.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=loader)[args.dataset]
        kb_entry_file = 'data/{}/feateng_data/kb_entry.pkl'.format(args.dataset)
        kb_file = 'data/{}/feateng_data/kb_inner_product.pkl'.format(args.dataset)

        # kb_entry_file = 'data/{}/feateng_data/kb_entry_wohist.pkl'.format(args.dataset)
        # kb_file = 'data/{}/feateng_data/kb_inner_product_wohist.pkl'.format(args.dataset)

        build_inner_product_as_kb(model_config, data_config, kb_entry_file, kb_file, 
                                  batch_size=20000, gpu_num=1)
    
    elif args.kb_mode == 'encoder_vec':
        model_config_path = os.path.join('configs/model_configs', data_config['kb_encoder_model_name'] + '.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=loader)[args.dataset]
        kb_entry_file = 'data/{}/feateng_data/kb_entry.pkl'.format(args.dataset)
        kb_file = 'data/{}/feateng_data/kb_encoder_vec.pkl'.format(args.dataset)

        # kb_entry_file = 'data/{}/feateng_data/kb_entry_wohist.pkl'.format(args.dataset)
        # kb_file = 'data/{}/feateng_data/kb_encoder_vec_wohist.pkl'.format(args.dataset)
        build_encoder_vec_as_kb(model_config, data_config, kb_entry_file, kb_file, 
                                  batch_size=20000, gpu_num=0)
    
    elif args.kb_mode == 'user_seq_rep':
        model_config_path = os.path.join('configs/model_configs', data_config['user_seq_encoder_model_name'] + '.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=loader)[args.dataset]
        kb_file = 'data/{}/feateng_data/kb_user_seq_rep.pkl'.format(args.dataset)
        data_path = 'data/{}/feateng_data/dataset'.format(args.dataset)
        build_user_seq_rep_as_kb(model_config, data_config, data_path, kb_blk_list,
                            kb_file, batch_size=20000, gpu_num=0)
