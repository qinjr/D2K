import pickle as pkl
import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm
from utils.yaml_loader import get_yaml_loader


def retrieve_klg4dataset_scalar_wohist(kb_file, dataset_files, klg_files, data_config):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
        print('kb loaded')
    for dataset_file, klg_file in zip(dataset_files, klg_files):
        print('start querying for dataset file: {}'.format(dataset_file))
        with open(dataset_file, 'rb') as f:
            data = pkl.load(f)
        x_user, x_item, x_context = data[0], data[1], data[2]
        klg = []
        for i in data_config['klg_user_feats']:
            for j in data_config['klg_item_feats']:
                for k in data_config['klg_context_feats']:
                    print('begin querying user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                    weights_ijk = []

                    user_feat_i = x_user[:,i].tolist()
                    item_feat_j = x_item[:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()
                    
                    for tup in tqdm(zip(user_feat_i, item_feat_j, context_feat_k)):
                        if tup in kb:
                            weights_ijk.append(kb[tup])
                        else:
                            weights_ijk.append(NO_RELATION_WEIGHT)
                    weights_ijk = np.array(weights_ijk).reshape([-1,1])
                    klg.append(weights_ijk)
        klg = np.concatenate(klg, axis=1)
        print('knowledge matrix shape is:', klg.shape)
        print('user matrix shape is:', x_user.shape)

        with open(klg_file, 'wb') as f:
            pkl.dump(klg, f)


def retrieve_klg4dataset_scalar(kb_file, dataset_files, klg_files, data_config):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
        print('kb loaded')
    for dataset_file, klg_file in zip(dataset_files, klg_files):
        print('start querying for dataset file: {}'.format(dataset_file))
        with open(dataset_file, 'rb') as f:
            x_user, x_item, x_context, _, hist, hist_len = pkl.load(f)
        klg = []
        for i in data_config['klg_user_feats']:
            for j in data_config['klg_item_feats']:
                for k in data_config['klg_context_feats']:
                    print('begin querying user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                    weights_ijk = []

                    user_feat_i = x_user[:,i].tolist()
                    item_feat_j = x_item[:,j].tolist()
                    context_feat_k = x_item[:,k].tolist()
                    
                    for tup in tqdm(zip(user_feat_i, item_feat_j, context_feat_k)):
                        if tup in kb:
                            weights_ijk.append(kb[tup])
                        else:
                            weights_ijk.append(NO_RELATION_WEIGHT)
                    weights_ijk = np.array(weights_ijk).reshape([-1,1])
                    klg.append(weights_ijk)
        for i in data_config['klg_item_feats']:
            for j in data_config['klg_hist_feats']:
                for k in data_config['klg_context_feats']:
                    print('begin querying item feat {} + hist feat {} + context feat {} in the KB'.format(i, j, k))
                    weights_ijk = []

                    item_feat_i = x_item[:,i].tolist()
                    user_seq_feat_j = hist[:,:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()

                    for p, (item_feat_i_p, context_feat_k_p) in tqdm(enumerate(zip(item_feat_i, context_feat_k))):
                        length = hist_len[p]
                        if length == 0:
                            weights_ijk.append(NO_RELATION_WEIGHT_HIST)
                            continue

                        user_seq_feat_j_p = user_seq_feat_j[p]
                        weights_ijk_pq = []
                        for q in range(length):
                            tup = (item_feat_i_p, user_seq_feat_j_p[q], context_feat_k_p)
                            if tup in kb:
                                weights_ijk_pq.append(kb[tup])
                            else:
                                weights_ijk_pq.append(NO_RELATION_WEIGHT_HIST)
                        weights_ijk.append(sum(weights_ijk_pq)/length)
                    weights_ijk = np.array(weights_ijk).reshape([-1,1])
                    klg.append(weights_ijk)
    
        klg = np.concatenate(klg, axis=1)
        print('knowledge matrix shape is:', klg.shape)
        print('user matrix shape is:', x_user.shape)

        with open(klg_file, 'wb') as f:
            pkl.dump(klg, f)

def retrieve_klg4dataset_vec_wohist(kb_file, dataset_files, klg_files, data_config):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
        print('kb loaded')
    for dataset_file, klg_file in zip(dataset_files, klg_files):
        print('start querying for dataset file: {}'.format(dataset_file))
        with open(dataset_file, 'rb') as f:
            data = pkl.load(f)
        x_user, x_item, x_context = data[0], data[1], data[2]
        klg = []
        for i in data_config['klg_user_feats']:
            for j in data_config['klg_item_feats']:
                for k in data_config['klg_context_feats']:
                    print('begin querying user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                    vecs_ijk = []

                    user_feat_i = x_user[:,i].tolist()
                    item_feat_j = x_item[:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()
                    for tup in tqdm(zip(user_feat_i, item_feat_j, context_feat_k)):
                        if tup in kb:
                            vecs_ijk.append(kb[tup])
                        else:
                            vecs_ijk.append([NO_RELATION_WEIGHT] * data_config['klg_encoder_vec_dim'])
                    vecs_ijk = np.array(vecs_ijk)
                    klg.append(vecs_ijk)
        klg = np.concatenate(klg, axis=1)
        print('knowledge matrix shape is:', klg.shape)
        print('user matrix shape is:', x_user.shape)

        with open(klg_file, 'wb') as f:
            pkl.dump(klg, f)

def retrieve_klg4dataset_vec(kb_file, dataset_files, klg_files, data_config):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
        print('kb loaded')
    for dataset_file, klg_file in zip(dataset_files, klg_files):
        print('start querying for dataset file: {}'.format(dataset_file))
        with open(dataset_file, 'rb') as f:
            x_user, x_item, x_context, _, hist, hist_len = pkl.load(f)
        klg = []
        for i in data_config['klg_user_feats']:
            for j in data_config['klg_item_feats']:
                for k in data_config['klg_context_feats']:
                    print('begin querying user feat {} + item feat {} + context feat {} in the KB'.format(i, j, k))
                    vecs_ijk = []

                    user_feat_i = x_user[:,i].tolist()
                    item_feat_j = x_item[:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()

                    for tup in tqdm(zip(user_feat_i, item_feat_j, context_feat_k)):
                        if tup in kb:
                            vecs_ijk.append(kb[tup])
                        else:
                            vecs_ijk.append([NO_RELATION_WEIGHT] * data_config['klg_encoder_vec_dim'])
                    vecs_ijk = np.array(vecs_ijk)
                    klg.append(vecs_ijk)
        for i in data_config['klg_item_feats']:
            for j in data_config['klg_hist_feats']:
                for k in data_config['klg_context_feats']:
                    print('begin querying item feat {} + hist feat {} + context feat {} in the KB'.format(i, j, k))
                    vecs_ijk = []

                    item_feat_i = x_item[:,i].tolist()
                    user_seq_feat_j = hist[:,:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()

                    for p, (item_feat_i_p, context_feat_k_p) in tqdm(enumerate(zip(item_feat_i, context_feat_k))):
                        length = hist_len[p]
                        if length == 0:
                            vecs_ijk.append([NO_RELATION_WEIGHT_HIST] * data_config['klg_encoder_vec_dim'])
                            continue

                        user_seq_feat_j_p = user_seq_feat_j[p]
                        vecs_ijk_pq = []
                        for q in range(length):
                            tup = (item_feat_i_p, user_seq_feat_j_p[q], context_feat_k_p)
                            if tup in kb:
                                vecs_ijk_pq.append(kb[tup])
                            else:
                                vecs_ijk_pq.append([NO_RELATION_WEIGHT_HIST] * data_config['klg_encoder_vec_dim'])
                        vecs_ijk_pq = np.mean(np.array(vecs_ijk_pq), axis=0)
                        vecs_ijk.append(vecs_ijk_pq)
                    vecs_ijk = np.array(vecs_ijk)
                    klg.append(vecs_ijk)
        
        klg = np.concatenate(klg, axis=1)
        print('knowledge matrix shape is:', klg.shape)
        print('user matrix shape is:', x_user.shape)

        with open(klg_file, 'wb') as f:
            pkl.dump(klg, f)

def retrieve_stat(kb_file, dataset_files):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
        print('kb loaded')
    print('num of kb entries: {}'.format(len(kb)))
    
    ratios_feature = []
    ratios_sample = []
    for dataset_file in dataset_files:
        print('start checking for dataset file: {}'.format(dataset_file))
        with open(dataset_file, 'rb') as f:
            x_user, x_item, x_context, _, hist, hist_len = pkl.load(f)
        
        total_queries = 0
        no_relation_cnt_per_feature = 0

        no_relation_cnt_per_sample = np.zeros([x_user.shape[0],])
        for i in data_config['klg_user_feats']:
            for j in data_config['klg_item_feats']:
                for k in data_config['klg_context_feats']:
                    user_feat_i = x_user[:,i].tolist()
                    item_feat_j = x_item[:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()

                    for p, tup in enumerate(zip(user_feat_i, item_feat_j, context_feat_k)):
                        total_queries += 1
                        if tup not in kb:
                            no_relation_cnt_per_feature += 1
                            no_relation_cnt_per_sample[p] += 1
        
        for i in data_config['klg_item_feats']:
            for j in data_config['klg_hist_feats']:
                for k in data_config['klg_context_feats']:
                    item_feat_i = x_item[:,i].tolist()
                    user_seq_feat_j = hist[:,:,j].tolist()
                    context_feat_k = x_context[:,k].tolist()
                    for p, (item_feat_i_p, context_feat_k_p) in enumerate(zip(item_feat_i, context_feat_k)):
                        length = hist_len[p]
                        total_queries += 1
                        if length == 0:
                            no_relation_cnt_per_feature += 1
                            no_relation_cnt_per_sample[p] += 1
                            continue
                        
                        user_seq_feat_j_p = user_seq_feat_j[p]
                        no_found = 0
                        for q in range(length):
                            tup = (item_feat_i_p, user_seq_feat_j_p[q], context_feat_k_p)
                            if tup not in kb:
                                no_found += 1
                        if no_found == length:
                            no_relation_cnt_per_feature += 1
                            no_relation_cnt_per_sample[p] += 1

        ratios_feature.append(no_relation_cnt_per_feature / total_queries)
        ratios_sample.append(np.mean(no_relation_cnt_per_sample))
        print('the ratio of no relation per feature and per sample in KB for dataset {} are {} and {}'.format(dataset_file, no_relation_cnt_per_feature / total_queries, np.mean(no_relation_cnt_per_sample)))
    print(ratios_feature)
    print(ratios_sample)

def retrieve_user_seq_klg(kb_file, dataset_files, klg_files, data_config):
    with open(kb_file, 'rb') as f:
        kb = pkl.load(f)
        print('kb loaded')
    for dataset_file, klg_file in zip(dataset_files, klg_files):
        total_queries = 0
        non_hit_num = 0

        print('start querying for dataset file: {}'.format(dataset_file))
        with open(dataset_file, 'rb') as f:
            klg = []
            x_user, x_item, x_context, _, hist, hist_len = pkl.load(f)
            uids = x_user[:,0].tolist()
            for uid in tqdm(uids):
                total_queries += 1
                if uid in kb:
                    klg.append(kb[uid])
                else:
                    klg.append([0] * data_config['klg_user_seq_dim'])
                    non_hit_num += 1
            klg = np.array(klg)
            print('klg shape: {}'.format(klg.shape))
            print('x_user shape: {}'.format(x_user.shape))
            print('non hit rate: {}'.format(non_hit_num / total_queries))

        with open(klg_file, 'wb') as f:
            pkl.dump(klg, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='ad')
    parser.add_argument('-q', '--query_files', type=str, help='query files to search the kb', default='7')
    parser.add_argument('-m', '--kb_mode', type=str, help='kb mode: (pos_ratio, inner_product, encoder_vec)', default='pos_ratio')
    args = parser.parse_args()

    # for AD & Tenrec
    if args.dataset in ['ad', 'tenrec']:
        NO_RELATION_WEIGHT = -1
        NO_RELATION_WEIGHT_HIST = -0.01 # For hist features, more smooth
    elif args.dataset in ['eleme']:
        # for Eleme
        NO_RELATION_WEIGHT = 0 #-1
        NO_RELATION_WEIGHT_HIST = 0 #-0.01 # For hist features, more smooth

    query_blk_list = list(map(int, args.query_files.split(',')))

    root_path = '..'
    os.chdir(root_path)
    data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    loader = get_yaml_loader()
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=loader)

    
    dataset_file_prefix = 'data/{}/feateng_data/dataset_'.format(args.dataset)
    dataset_files = [dataset_file_prefix + '{}.pkl'.format(i) for i in query_blk_list]
    print(dataset_files)

    if args.kb_mode == 'pos_ratio':
        retrieve_stat('data/{}/feateng_data/kb_pos_ratio.pkl'.format(args.dataset), dataset_files)

    if args.kb_mode == 'pos_ratio':
        kb_file = 'data/{}/feateng_data/kb_pos_ratio.pkl'.format(args.dataset)
        klg_file_prefix = 'data/{}/feateng_data/klg_pos_ratio_'.format(args.dataset)
        
        # kb_file = 'data/{}/feateng_data/kb_pos_ratio_wohist.pkl'.format(args.dataset)
        # klg_file_prefix = 'data/{}/feateng_data/klg_pos_ratio_wohist_'.format(args.dataset)
        
        klg_files = [klg_file_prefix + '{}.pkl'.format(i) for i in query_blk_list]
        print(klg_files)

        retrieve_klg4dataset_scalar(kb_file, dataset_files, klg_files, data_config)
        # retrieve_klg4dataset_scalar_wohist(kb_file, dataset_files, klg_files, data_config)

    elif args.kb_mode == 'inner_product':
        kb_file = 'data/{}/feateng_data/kb_inner_product.pkl'.format(args.dataset)
        klg_file_prefix = 'data/{}/feateng_data/klg_inner_product_'.format(args.dataset)

        # kb_file = 'data/{}/feateng_data/kb_inner_product_wohist.pkl'.format(args.dataset)
        # klg_file_prefix = 'data/{}/feateng_data/klg_inner_product_wohist_'.format(args.dataset)
        klg_files = [klg_file_prefix + '{}.pkl'.format(i) for i in query_blk_list]
        print(klg_files)

        retrieve_klg4dataset_scalar(kb_file, dataset_files, klg_files, data_config)
        # retrieve_klg4dataset_scalar_wohist(kb_file, dataset_files, klg_files, data_config)
    elif args.kb_mode == 'encoder_vec':
        kb_file = 'data/{}/feateng_data/kb_encoder_vec.pkl'.format(args.dataset)
        klg_file_prefix = 'data/{}/feateng_data/klg_encoder_vec_'.format(args.dataset)

        # kb_file = 'data/{}/feateng_data/kb_encoder_vec_wohist.pkl'.format(args.dataset)
        # klg_file_prefix = 'data/{}/feateng_data/klg_encoder_vec_wohist_'.format(args.dataset)
        klg_files = [klg_file_prefix + '{}.pkl'.format(i) for i in query_blk_list]
        print(klg_files)
        
        retrieve_klg4dataset_vec(kb_file, dataset_files, klg_files, data_config)
        # retrieve_klg4dataset_vec_wohist(kb_file, dataset_files, klg_files, data_config)
    elif args.kb_mode == 'user_seq_rep':
        kb_file = 'data/{}/feateng_data/kb_user_seq_rep.pkl'.format(args.dataset)
        klg_file_prefix = 'data/{}/feateng_data/klg_user_seq_rep_'.format(args.dataset)
        klg_files = [klg_file_prefix + '{}.pkl'.format(i) for i in query_blk_list]
        
        retrieve_user_seq_klg(kb_file, dataset_files, klg_files, data_config)