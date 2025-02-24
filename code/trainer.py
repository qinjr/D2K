import sys
import os
from logging import getLogger
from time import time
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from recbole.utils import set_color

from utils.evaluations import TopKMetric, PointMetric
from utils.early_stopper import EarlyStopper
from utils.log import dict2str, get_tensorboard, get_local_time, ensure_dir, combo_dict
from loss import BPRLoss, RegLoss, SoftmaxLoss

class AbsTrainer(object):
    r"""Independent Trainer Class is used to manage the training and evaluation processes of recommender system models in independent training mode.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model, dataset_name):
        self.config = config
        self.model = model
        self.dataset_name = dataset_name

    def fit(self):
        r"""Train the model based on the train data.
        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self):
        r"""Evaluate the model based on the eval data.
        """

        raise NotImplementedError('Method [next] should be implemented.')

class Trainer(AbsTrainer):
    r"""Independent trainer for the warmup training phase
    """
    def __init__(self, config, model, dataset_name):
        super().__init__(config, model, dataset_name)
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.train_mode = config['train_mode']
        if self.model.batch_random_neg:
            self.train_mode = 'batch_neg'
        self.batch_neg_size = config['batch_neg_size']
        self.loss_type = config['loss_type']
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.l2_norm = config['l2_norm']
        self.max_epochs = config['max_epochs']

        # how much training epoch will we conduct one evaluation
        self.eval_step = min(config['eval_step'], self.max_epochs)
        self.clip_grad_norm = config['clip_grad_norm']

        self.eval_batch_size = config['eval_batch_size']
        self.device = torch.device(config['device'])
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}-{}.pth'.format(self.model.get_name().lower(), dataset_name, get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.continue_metric = config['continue_metric']
        self.best_eval_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer(self.model.parameters())

        # for eval
        self.eval_mode = config['eval_mode']
        self.list_len = config['list_len']
        self.topks = config['topks']

    def _build_optimizer(self, params):
        r"""Init the Optimizer
        Returns:
            torch.optim: the optimizer
        """
        # if self.config['reg_weight'] and self.weight_decay and self.weight_decay * self.config['reg_weight'] > 0:
        #     self.logger.warning(
        #         'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
        #         'which may lead to double regularization.'
        #     )
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _get_loss_func(self, loss_type):
        if loss_type == 'll':
            return torch.nn.BCELoss()
        elif loss_type == 'bpr':
            return BPRLoss()
        elif loss_type == 'reg':
            return RegLoss()
        elif loss_type == 'softmax':
            return SoftmaxLoss()
        elif loss_type == 'kd':
            return torch.nn.BCELoss()
    
    def _get_user_hist(self, x_user, stage = 'train'):
        uids = x_user[:,0].tolist()
        user_history = []
        hist_len = []
        if stage == 'train':
            for uid in uids:
                user_history.append(self.hist_dict_train[uid])
                hist_len.append(self.hist_len_dict_train[uid])
        elif stage == 'test':
            for uid in uids:
                user_history.append(self.hist_dict_test[uid])
                hist_len.append(self.hist_len_dict_test[uid])
        return torch.tensor(user_history), torch.tensor(hist_len)
    

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.
        Args:
            epoch (int): the current epoch id
        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_eval_result': self.best_eval_result,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.
        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_eval_result = checkpoint['best_eval_result']

        # load architecture params from checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)
    
    def load_model(self, resume_file):
        r"""Load the model parameters information and training information.
        Args:
            resume_file (file): the checkpoint file
        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file, map_location=self.device)

        # load architecture params from checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])

        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)
        return self.model
    
    def self_resume_checkpoint(self):
        resume_file = str(self.saved_model_file)
        checkpoint = torch.load(resume_file, map_location=self.device)
        # self.start_epoch = checkpoint['epoch'] + 1
        # self.cur_step = checkpoint['cur_step']
        # self.best_eval_result = checkpoint['best_eval_result']

        # load architecture params from checkpoint: JUST LOAD THE MODEL PARAMS
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def load_embedding_from_pretrain_model(self, resume_file):
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file, map_location=self.device)
        pretrained_embedding = checkpoint['state_dict']['embedding.embedding.weight']
        self.model.embedding.embedding.from_pretrained(pretrained_embedding, freeze=False)
        self.logger.info('Pretrained embedding matrix from: {} has beed loaded'.format(resume_file))

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        des = 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        
        des = '%.' + str(des) + 'f'
        train_loss_output += set_color('train loss', 'blue') + ': ' + des % loss
        return train_loss_output + ']'

    def _train_epoch(self, train_dl, epoch_idx, show_progress=True):
        self.model.train()
        main_loss_func = self._get_loss_func(self.loss_type)
        if self.train_mode == 'batch_neg':
            main_loss_func = self._get_loss_func('softmax')
        reg_loss_func = self._get_loss_func('reg')
        total_loss = None
        iter_data = (
            tqdm(
                train_dl,
                total=len(train_dl),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else train_dl
        )
        # preds = None
        # labels = None

        for batch_idx, batch_data in enumerate(iter_data):
            self.optimizer.zero_grad()
            if self.train_mode == 'point':
                x_user, x_item, x_context, y, user_hist, hist_len, klg = batch_data
                x_user = x_user.to(self.device)
                x_item = x_item.to(self.device)
                x_context = x_context.to(self.device)
                y = y.float().to(self.device)
                if self.model.use_hist:
                    user_hist = user_hist.to(self.device)
                    hist_len = hist_len.to(self.device)
                if self.model.use_klg:
                    klg = klg.float().to(self.device)                    
                    

                pred = self.model(x_user, x_item, x_context, user_hist, hist_len, klg)
                loss = main_loss_func(pred, y) + self.l2_norm * reg_loss_func(self.model.parameters())
            else:
                print('unsupported train mode: {}'.format(self.train_mode))
                exit(1)

            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

        return total_loss / len(train_dl)
    
    def _train_epoch_kd(self, teacher_model, train_dl, epoch_idx, show_progress=True):
        self.model.train()
        main_loss_func = self._get_loss_func(self.loss_type)
        if self.train_mode == 'batch_neg':
            main_loss_func = self._get_loss_func('softmax')
        reg_loss_func = self._get_loss_func('reg')
        kd_loss_func = self._get_loss_func('kd')
        total_loss = None
        iter_data = (
            tqdm(
                train_dl,
                total=len(train_dl),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else train_dl
        )
        # preds = None
        # labels = None

        for batch_idx, batch_data in enumerate(iter_data):
            self.optimizer.zero_grad()
            if self.train_mode == 'point':
                x_user, x_item, x_context, y, user_hist, hist_len, klg = batch_data
                x_user = x_user.to(self.device)
                x_item = x_item.to(self.device)
                x_context = x_context.to(self.device)
                y = y.float().to(self.device)
                if self.model.use_hist:
                    user_hist = user_hist.to(self.device)
                    hist_len = hist_len.to(self.device)
                if self.model.use_klg:
                    klg = klg.float().to(self.device)                    
                    

                pred = self.model(x_user, x_item, x_context, user_hist, hist_len, klg)
                teacher_pred = teacher_model(x_user, x_item, x_context, user_hist, hist_len, klg)

                loss = main_loss_func(pred, y) + kd_loss_func(pred, teacher_pred)
            else:
                print('unsupported train mode: {}'.format(self.train_mode))
                exit(1)

            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

        return total_loss / len(train_dl)

    def _train_epoch_with_hm(self, hist_models, train_dl, epoch_idx, show_progress=True):
        self.model.train()
        main_loss_func = self._get_loss_func(self.loss_type)
        if self.train_mode == 'batch_neg':
            main_loss_func = self._get_loss_func('softmax')
        reg_loss_func = self._get_loss_func('reg')
        total_loss = None
        iter_data = (
            tqdm(
                train_dl,
                total=len(train_dl),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else train_dl
        )
        # preds = None
        # labels = None

        for batch_idx, batch_data in enumerate(iter_data):
            self.optimizer.zero_grad()
            if self.train_mode == 'point':
                x_user, x_item, x_context, y, user_hist, hist_len, klg = batch_data
                user_hist_emb = None

                x_user = x_user.to(self.device)
                x_user_emb = self.model.get_feat_embedding(x_user)
                
                x_item = x_item.to(self.device)
                x_item_emb = self.model.get_feat_embedding(x_item)

                x_context = x_context.to(self.device)
                x_context_emb = self.model.get_feat_embedding(x_context)

                y = y.float().to(self.device)
                if self.model.use_hist:
                    user_hist = user_hist.to(self.device)
                    user_hist_emb = self.model.get_feat_embedding(user_hist)
                    hist_len = hist_len.to(self.device)
                if self.model.use_klg:
                    klg = klg.float().to(self.device)                    
                
                preds_by_hist_models = [hm(x_user_emb, x_item_emb, x_context_emb, user_hist_emb, hist_len, klg).unsqueeze(1) for hm in hist_models]
                preds_by_hist_models = torch.mean(torch.cat(preds_by_hist_models, dim=1), dim=1)

                pred = 0.5 * (self.model(x_user, x_item, x_context, user_hist, hist_len, klg) + preds_by_hist_models)
                loss = main_loss_func(pred, y) + self.l2_norm * reg_loss_func(self.model.parameters())
            else:
                print('unsupported train mode: {}'.format(self.train_mode))
                exit(1)

            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

        return total_loss / len(train_dl)


    @torch.no_grad()
    def evaluate(self, dataloader, load_best_model=True, model_file=None, show_progress=False):
        if not dataloader:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        # preds = None
        # labels = None
        preds = []
        labels = []
        
        # collect data
        for batch_idx, batch_data in enumerate(iter_data):
            x_user, x_item, x_context, y, user_hist, hist_len, klg = batch_data
            x_user = x_user.to(self.device)
            x_item = x_item.to(self.device)
            x_context = x_context.to(self.device)
            y = y.float().to(self.device)
            if self.model.use_hist:
                user_hist = user_hist.to(self.device)
                hist_len = hist_len.to(self.device)
            if self.model.use_klg:
                klg = klg.float().to(self.device)                

            pred = self.model(x_user, x_item, x_context, user_hist, hist_len, klg)

            preds.append(pred) #pred if preds is None else torch.cat((preds, pred))
            labels.append(y) #y if labels is None else torch.cat((labels, y))

        preds = torch.cat(preds).cpu().detach().numpy()
        labels = torch.cat(labels).cpu().detach().numpy()

        # get metrics
        metrics_point = PointMetric(labels, preds)
        eval_result_point = metrics_point.get_metrics()

        if self.config['eval_mode'] in ['all', 'list']:
            metrics_topk = TopKMetric(self.topks, self.list_len, labels, preds, None)
            eval_result_topk = metrics_topk.get_metrics()
            res = combo_dict(eval_result_topk, eval_result_point) if self.config['eval_mode'] == 'all' else eval_result_topk
        elif self.config['eval_mode'] == 'point':
            res = eval_result_point

        return res, preds, labels

    @torch.no_grad()
    def evaluate_with_hm(self, hist_models, dataloader, load_best_model=True, model_file=None, show_progress=False):
        if not dataloader:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        iter_data = (
            tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
                position=0, 
                leave=True
            ) if show_progress else dataloader
        )

        # preds = None
        # labels = None
        preds = []
        labels = []
        
        # collect data
        for batch_idx, batch_data in enumerate(iter_data):
            x_user, x_item, x_context, y, user_hist, hist_len, klg = batch_data
            user_hist_emb = None

            x_user = x_user.to(self.device)
            x_user_emb = self.model.get_feat_embedding(x_user)
            
            x_item = x_item.to(self.device)
            x_item_emb = self.model.get_feat_embedding(x_item)

            x_context = x_context.to(self.device)
            x_context_emb = self.model.get_feat_embedding(x_context)

            y = y.float().to(self.device)
            if self.model.use_hist:
                user_hist = user_hist.to(self.device)
                user_hist_emb = self.model.get_feat_embedding(user_hist)
                hist_len = hist_len.to(self.device)
            if self.model.use_klg:
                klg = klg.float().to(self.device)   

            preds_by_hist_models = [hm(x_user_emb, x_item_emb, x_context_emb, user_hist_emb, hist_len, klg).unsqueeze(1) for hm in hist_models]
            preds_by_hist_models = torch.mean(torch.cat(preds_by_hist_models, dim=1), dim=1)

            pred = 0.5 * (self.model(x_user, x_item, x_context, user_hist, hist_len, klg) + preds_by_hist_models)

            preds.append(pred) #pred if preds is None else torch.cat((preds, pred))
            labels.append(y) #y if labels is None else torch.cat((labels, y))

        preds = torch.cat(preds).cpu().detach().numpy()
        labels = torch.cat(labels).cpu().detach().numpy()

        # get metrics
        metrics_point = PointMetric(labels, preds)
        eval_result_point = metrics_point.get_metrics()

        if self.config['eval_mode'] in ['all', 'list']:
            metrics_topk = TopKMetric(self.topks, self.list_len, labels, preds, None)
            eval_result_topk = metrics_topk.get_metrics()
            res = combo_dict(eval_result_topk, eval_result_point) if self.config['eval_mode'] == 'all' else eval_result_topk
        elif self.config['eval_mode'] == 'point':
            res = eval_result_point

        return res, preds, labels

    def fit(self, train_dl, test_dl=None, verbose=True, saved=True, show_progress=True):
        train_dl.refresh()
        test_dl.refresh()
        
        if saved and self.start_epoch >= self.max_epochs:
            self._save_checkpoint(-1)
        
        early_stopper = EarlyStopper(self.config)
        
        eval_result, _, _ = self.evaluate(test_dl, False, show_progress=True)
        eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)
        test_dl.refresh()
        self.logger.info(eval_output)

        eval_results = []
        for epoch_idx in range(self.start_epoch, self.max_epochs):
            if epoch_idx > self.start_epoch:
                train_dl.refresh()
                test_dl.refresh()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_dl, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self.tensorboard.add_scalar('Loss/Train', train_loss, epoch_idx)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                eval_start_time = time()
                eval_result, _, _ = self.evaluate(test_dl, False, show_progress=True)
                eval_end_time = time()
                eval_results.append(eval_result)
                
                continue_metric_value = eval_result[self.continue_metric]
                continue_metric_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color(self.continue_metric, 'blue') + ": %f]") % \
                                     (epoch_idx, eval_end_time - eval_start_time, continue_metric_value)
                eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)

                if verbose:
                    self.logger.info(continue_metric_output)
                    self.logger.info(eval_output)
                self.tensorboard.add_scalar('eval_{}'.format(self.continue_metric), continue_metric_value, epoch_idx)
                continue_flag, save_flag = early_stopper.check(continue_metric_value, epoch_idx)
                if epoch_idx == 0:
                    save_flag = True

                if save_flag and saved:
                    self._save_checkpoint(epoch_idx)
                    save_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(save_output)

                if not continue_flag:
                    break
        
        best_epoch = early_stopper.get_best_epoch_idx()
        best_eval_result = eval_results[best_epoch]
        eval_output = set_color('best eval result', 'blue') + ': \n' + dict2str(best_eval_result)
        self.logger.info(eval_output)

        return best_eval_result

    def kd_fit(self, teacher_model, train_dl, test_dl=None, verbose=True, saved=True, show_progress=True):
        train_dl.refresh()
        test_dl.refresh()
        
        if saved and self.start_epoch >= self.max_epochs:
            self._save_checkpoint(-1)
        
        early_stopper = EarlyStopper(self.config)
        
        eval_result, _, _ = self.evaluate(test_dl, False, show_progress=True)
        eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)
        test_dl.refresh()
        self.logger.info(eval_output)

        eval_results = []
        for epoch_idx in range(self.start_epoch, self.max_epochs):
            if epoch_idx > self.start_epoch:
                train_dl.refresh()
                test_dl.refresh()
            # train
            training_start_time = time()
            train_loss = self._train_epoch_kd(teacher_model, train_dl, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self.tensorboard.add_scalar('Loss/Train', train_loss, epoch_idx)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                eval_start_time = time()
                eval_result, _, _ = self.evaluate(test_dl, False, show_progress=True)
                eval_end_time = time()
                eval_results.append(eval_result)
                
                continue_metric_value = eval_result[self.continue_metric]
                continue_metric_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color(self.continue_metric, 'blue') + ": %f]") % \
                                     (epoch_idx, eval_end_time - eval_start_time, continue_metric_value)
                eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)

                if verbose:
                    self.logger.info(continue_metric_output)
                    self.logger.info(eval_output)
                self.tensorboard.add_scalar('eval_{}'.format(self.continue_metric), continue_metric_value, epoch_idx)
                continue_flag, save_flag = early_stopper.check(continue_metric_value, epoch_idx)
                if epoch_idx == 0:
                    save_flag = True

                if save_flag and saved:
                    self._save_checkpoint(epoch_idx)
                    save_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(save_output)

                if not continue_flag:
                    break
        
        best_epoch = early_stopper.get_best_epoch_idx()
        best_eval_result = eval_results[best_epoch]
        eval_output = set_color('best eval result', 'blue') + ': \n' + dict2str(best_eval_result)
        self.logger.info(eval_output)

        return best_eval_result

    def fit_with_hm(self, hist_models, train_dl, test_dl=None, verbose=True, saved=True, show_progress=True):
        train_dl.refresh()
        test_dl.refresh()
        
        if saved and self.start_epoch >= self.max_epochs:
            self._save_checkpoint(-1)
        
        early_stopper = EarlyStopper(self.config)
        
        eval_result, _, _ = self.evaluate_with_hm(hist_models, test_dl, False, show_progress=True)
        eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)
        test_dl.refresh()
        self.logger.info(eval_output)

        eval_results = []
        for epoch_idx in range(self.start_epoch, self.max_epochs):
            if epoch_idx > self.start_epoch:
                train_dl.refresh()
                test_dl.refresh()
            # train
            training_start_time = time()
            train_loss = self._train_epoch_with_hm(hist_models, train_dl, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self.tensorboard.add_scalar('Loss/Train', train_loss, epoch_idx)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                eval_start_time = time()
                eval_result, _, _ = self.evaluate_with_hm(hist_models, test_dl, False, show_progress=True)
                eval_end_time = time()
                eval_results.append(eval_result)
                
                continue_metric_value = eval_result[self.continue_metric]
                continue_metric_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color(self.continue_metric, 'blue') + ": %f]") % \
                                     (epoch_idx, eval_end_time - eval_start_time, continue_metric_value)
                eval_output = set_color('eval result', 'blue') + ': \n' + dict2str(eval_result)

                if verbose:
                    self.logger.info(continue_metric_output)
                    self.logger.info(eval_output)
                self.tensorboard.add_scalar('eval_{}'.format(self.continue_metric), continue_metric_value, epoch_idx)
                continue_flag, save_flag = early_stopper.check(continue_metric_value, epoch_idx)
                if epoch_idx == 0:
                    save_flag = True

                if save_flag and saved:
                    self._save_checkpoint(epoch_idx)
                    save_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(save_output)

                if not continue_flag:
                    break
        
        best_epoch = early_stopper.get_best_epoch_idx()
        best_eval_result = eval_results[best_epoch]
        eval_output = set_color('best eval result', 'blue') + ': \n' + dict2str(best_eval_result)
        self.logger.info(eval_output)

        return best_eval_result

