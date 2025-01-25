import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import pickle as pkl
import torch

class Dataloader(object):
    def __init__(self, data_path, split_idx: list, batch_size, shuffle) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

        dataset_tuples_buf = []
        # loading split datasets
        for i in split_idx:
            dataset_path = data_path + '_{}.pkl'.format(i)
            with open(dataset_path, 'rb') as f:
                dataset_tuple_i = pkl.load(f)
                dataset_tuples_buf.append(dataset_tuple_i)
        concat_buf = []
        dataset_tuple_size = len(dataset_tuples_buf[0])
        for i in range(dataset_tuple_size):
            concat_buf.append([dt[i] for dt in dataset_tuples_buf])

        self.dataset_tuple = [np.concatenate(concat, axis=0) for concat in concat_buf]
        self.dataset_tuple = [torch.from_numpy(t) for t in self.dataset_tuple]

        self.dataset_size = len(self.dataset_tuple[0])
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
        # if self.shuffle:
        #     self._shuffle_data()
    
    def __len__(self):
        return self.total_step

    def _shuffle_data(self):
        print('shuffling...')
        perm = np.random.permutation(self.dataset_size)
        for i in range(len(self.dataset_tuple)):
            self.dataset_tuple[i] = self.dataset_tuple[i][perm]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            right = self.dataset_size
        else:
            right = self.batch_size * (self.step + 1)
        
        self.step += 1
        batch_data = []
        for i in range(len(self.dataset_tuple)):
            batch_data.append(self.dataset_tuple[i][left:right])
        return batch_data + [None] # for consistant with klg

    def refresh(self):
        print('refreshing...')
        self.step = 0
        if self.shuffle:
            self._shuffle_data()
        print('refreshed')
    
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1


class DataloaderKLG(object):
    def __init__(self, data_path, klg_path, split_idx: list, batch_size, shuffle) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle

        dataset_tuples_buf = []
        # loading split datasets
        for i in split_idx:
            dataset_path = data_path + '_{}.pkl'.format(i)
            klg_path_i = klg_path + '_{}.pkl'.format(i)
            with open(dataset_path, 'rb') as f:
                dataset_tuple_i = pkl.load(f)
            if os.path.exists(klg_path_i):
                with open(klg_path_i, 'rb') as f:
                    klg_i = pkl.load(f)
                    dataset_tuple_i.append(klg_i)
            else:
                dataset_tuple_i.append(np.ones_like(dataset_tuple_i[0]))

            dataset_tuples_buf.append(dataset_tuple_i)
        concat_buf = []
        dataset_tuple_size = len(dataset_tuples_buf[0])
        for i in range(dataset_tuple_size):
            concat_buf.append([dt[i] for dt in dataset_tuples_buf])

        self.dataset_tuple = [np.concatenate(concat, axis=0) for concat in concat_buf]
        self.dataset_tuple = [torch.from_numpy(t) for t in self.dataset_tuple]

        self.dataset_size = len(self.dataset_tuple[0])
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
        # if self.shuffle:
        #     self._shuffle_data()
    
    def __len__(self):
        return self.total_step

    def _shuffle_data(self):
        print('shuffling...')
        perm = np.random.permutation(self.dataset_size)
        for i in range(len(self.dataset_tuple)):
            self.dataset_tuple[i] = self.dataset_tuple[i][perm]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            right = self.dataset_size
        else:
            right = self.batch_size * (self.step + 1)
        
        self.step += 1
        batch_data = []
        for i in range(len(self.dataset_tuple)):
            batch_data.append(self.dataset_tuple[i][left:right])
        return batch_data

    def refresh(self):
        print('refreshing...')
        self.step = 0
        if self.shuffle:
            self._shuffle_data()
        print('refreshed')
    
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1


class KBEntryloader(object):
    def __init__(self, batch_size, kb_entry_file):
        super().__init__()
        self.batch_size = batch_size

        with open(kb_entry_file, 'rb') as f:
            self.kb_entry = pkl.load(f)
        print(self.kb_entry.shape)
        
        self.dataset_size = len(self.kb_entry)
        if self.dataset_size % self.batch_size == 0:
            self.total_step = int(self.dataset_size / self.batch_size)
        else:
            self.total_step = int(self.dataset_size / self.batch_size) + 1
        self.step = 0
    
    def __len__(self):
        return self.total_step

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.total_step:
            raise StopIteration

        left = self.batch_size * self.step
        if self.step == self.total_step - 1:
            right = self.dataset_size
        else:
            right = self.batch_size * (self.step + 1)
        
        self.step += 1        
        return self.kb_entry[left:right]

    def refresh(self):
        self.step = 0
    


if __name__ == '__main__':
    dl = DataloaderKLG('../data/ad/feateng_data/dataset', '../data/ad/feateng_data/klg', [1], 256, True)
    for batch_data in tqdm(dl):
        x_user, x_item, y, hist, hist_len, klg = batch_data
        print(x_user.shape)
        print(x_item.shape)
        print(y.shape)
        print(hist.shape)
        print(hist_len.shape)
        print(klg.shape)
        print(klg)
        break
    
    # dl = KBEntryloader(1024, '../data/ad/feateng_data/kb_entry.pkl')
    # for batch_data in tqdm(dl):
    #     print(batch_data.shape)
    #     break