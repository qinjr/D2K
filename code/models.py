import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import FeaturesEmbedding
from layers import Linear
from layers import FactorizationLayer
from layers import MultiLayerPerceptron, MultiLayerPerceptron4Encoder
from layers import InnerProductNetwork, UIInnerProductNetwork, UICrossLayer, UICrossLayer2, IHCrossLayer, UICCrossLayer, IHCCrossLayer
from layers import HistAtt, HistAttProd
from recbole.model.init import xavier_normal_initialization

import logging

class Rec(torch.nn.Module):
    def __init__(self, model_config, data_config):
        super().__init__()
        self.vocabulary_size = data_config['vocabulary_size']
        self.user_num_fields = data_config['user_num_fields']
        self.item_num_fields = data_config['item_num_fields']
        self.context_num_fields = data_config['context_num_fields']
        self.hist_num_fields = data_config['hist_num_fields']
        self.x_num_fields = self.user_num_fields + self.item_num_fields + self.context_num_fields

        self.klg_hist_feats = data_config['klg_hist_feats']
        self.klg_user_feats = data_config['klg_user_feats']
        self.klg_item_feats = data_config['klg_item_feats']
        self.klg_context_feats = data_config['klg_context_feats']
        
        
        self.klg_num_fields = (len(self.klg_user_feats) + len(self.klg_hist_feats)) * len(self.klg_item_feats) * len(self.klg_context_feats) if data_config['klg_has_hist'] else len(self.klg_user_feats) * len(self.klg_item_feats) * len(self.klg_context_feats)
        
        self.klg_encoder_vec_dim = data_config['klg_encoder_vec_dim']
        self.embed_dim = model_config['embed_dim'] if 'embed_dim' in model_config else None
        self.hidden_dims = model_config['hidden_dims'] if 'hidden_dims' in model_config else None
        self.dropout = model_config['dropout'] if 'dropout' in model_config else None
        self.use_hist = model_config['use_hist'] if 'use_hist' in model_config else None
        self.use_klg = model_config['use_klg'] if 'use_klg' in model_config else None
        self.batch_random_neg = model_config['batch_random_neg'] if 'batch_random_neg' in model_config else None
        self.act = model_config['act'] if 'act' in model_config else None
        self.n_head = model_config['n_head'] if 'n_head' in model_config else None
        self.d_model = model_config['d_model'] if 'd_model' in model_config else None
        self.klg_usage_type = model_config['klg_usage_type'] if 'klg_usage_type' in model_config else None
        self.klg_mlp_hidden_dims = model_config['klg_mlp_hidden_dims'] if 'klg_mlp_hidden_dims' in model_config else None
        self.klg_process = model_config['klg_process'] if 'klg_process' in model_config else None
        self.klg_processer_mlp_layers = model_config['klg_processer_mlp_layers'] if 'klg_processer_mlp_layers' in model_config else None
        self.embed_dim2 = model_config['embed_dim2'] if 'embed_dim2' in model_config else None
        self.x_mlp_act = model_config['x_mlp_act'] if 'x_mlp_act' in model_config else None
        self.share_emb = model_config['share_emb'] if 'share_emb' in model_config else None


class ONLY_KLG_LR(Rec):
    def __init__(self, model_config, data_config, klg_type):
        super(ONLY_KLG_LR, self).__init__(model_config, data_config)
        if klg_type in ['pos_ratio', 'inner_product']:
            self.klg_vec_dim = self.klg_num_fields * 1
        elif klg_type in ['encoder_vec']:
            self.klg_vec_dim = self.klg_num_fields * self.klg_encoder_vec_dim
        self.linear = torch.nn.Linear(self.klg_vec_dim, 1)

        # for xmlp
        self.total_params_num = self.klg_processer_mlp_layers * (self.klg_encoder_vec_dim + 1) * self.klg_encoder_vec_dim
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim, self.total_params_num)
        if self.x_mlp_act == 'tanh':
            self.f_a = torch.nn.Tanh()
        elif self.x_mlp_act == 'relu':
            self.f_a = torch.nn.ReLU()
        

    def get_name(self):
        return 'ONLY_KLG_LR'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        # x = torch.cat((x_user, x_item, x_context), dim=1)
        # klg = self._process_klg_in_x_mlp(klg, x)
        return torch.sigmoid(self.linear(klg).squeeze(1))
    
    def _process_klg_in_x_mlp(self, klg, x):
        # get params
        emb2_x = self.embedding(x).view(-1, self.x_num_fields * self.embed_dim)
        
        params = self.x_projector(emb2_x)

        # weights and bias
        weights_list = []
        bias_list = []
        for i in range(self.klg_processer_mlp_layers):
            weight_lhs = i*(self.klg_encoder_vec_dim+1)*self.klg_encoder_vec_dim
            weight_rhs = weight_lhs + self.klg_encoder_vec_dim*self.klg_encoder_vec_dim
            weights_list.append(params[:,weight_lhs:weight_rhs].reshape((-1,self.klg_encoder_vec_dim,self.klg_encoder_vec_dim)))

            bias_lhs = (i+1)*self.klg_encoder_vec_dim*self.klg_encoder_vec_dim + i*self.klg_encoder_vec_dim
            bias_rhs = bias_lhs + self.klg_encoder_vec_dim
            bias_list.append(torch.unsqueeze(params[:,bias_lhs:bias_rhs],dim=1))
        
        # forward klg
        klg = klg.reshape([-1, self.klg_num_fields, self.klg_encoder_vec_dim])
        klg_x_mlp = klg
        for i in range(self.klg_processer_mlp_layers):
            klg_x_mlp = self.f_a(torch.matmul(klg_x_mlp, weights_list[i]) + bias_list[i])
        return klg_x_mlp.view(-1, self.klg_vec_dim)

class LR_KLG(Rec):
    def __init__(self, model_config, data_config, klg_type):
        super(LR_KLG, self).__init__(model_config, data_config)
        if klg_type in ['pos_ratio', 'inner_product']:
            self.klg_vec_dim = self.klg_num_fields * 1
        elif klg_type in ['encoder_vec']:
            self.klg_vec_dim = self.klg_num_fields * self.klg_encoder_vec_dim
        elif klg_type in ['user_seq_rep']:
            self.klg_vec_dim = self.hist_num_fields * self.embed_dim
        
        self.linear = Linear(self.vocabulary_size)
        if self.klg_usage_type in ['tower-lr', 'concat']:
            self.klg_predictor = torch.nn.Linear(self.klg_vec_dim, 1)
        elif self.klg_usage_type == 'tower-mlp':
            self.klg_predictor = MultiLayerPerceptron(self.klg_vec_dim, self.klg_mlp_hidden_dims, 0.2)


    def get_name(self):
        return 'LR_KLG'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        return torch.sigmoid(self.linear(x).squeeze(1) + self.klg_predictor(klg).squeeze(1))

class LR(Rec):
    def __init__(self, model_config, data_config):
        super(LR, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
    
    def get_name(self):
        return 'LR'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        return torch.sigmoid(self.linear(x).squeeze(1))

class FM(Rec):
    def __init__(self, model_config, data_config):
        super(FM, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        
    def get_name(self):
        return 'FM'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x)
        return torch.sigmoid(x.squeeze(1))
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

class DeepFM_KLG(Rec):
    def __init__(self, model_config, data_config, klg_type):
        super(DeepFM_KLG, self).__init__(model_config, data_config)
        if klg_type in ['pos_ratio', 'inner_product']:
            self.klg_vec_dim = self.klg_num_fields * 1
        elif klg_type in ['encoder_vec']:
            self.klg_vec_dim = self.klg_num_fields * self.klg_encoder_vec_dim
        elif klg_type in ['user_seq_rep']:
            self.klg_vec_dim = self.hist_num_fields * self.embed_dim
        
        self.linear = Linear(self.vocabulary_size)
        
        if self.klg_usage_type == 'tower-lr':
            self.klg_predictor = torch.nn.Linear(self.klg_vec_dim, 1)
        elif self.klg_usage_type == 'tower-mlp':
            self.klg_predictor = MultiLayerPerceptron(self.klg_vec_dim, self.klg_mlp_hidden_dims, self.dropout)

        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.x_num_fields * self.embed_dim
        if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
            self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)
        elif self.klg_usage_type == 'concat':
            self.mlp = MultiLayerPerceptron(self.input_dim + self.klg_vec_dim, self.hidden_dims, self.dropout)
        
        if self.klg_process == 'x_mlp':
            self.total_params_num = self.klg_processer_mlp_layers * (self.klg_encoder_vec_dim + 1) * self.klg_encoder_vec_dim
                
            if self.share_emb:
                self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim, self.total_params_num)
            else:
                self.embedding2 = FeaturesEmbedding(self.vocabulary_size, self.embed_dim2)
                self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim2, self.total_params_num)
            
            if self.x_mlp_act == 'tanh':
                self.f_a = torch.nn.Tanh()
            elif self.x_mlp_act == 'relu':
                self.f_a = torch.nn.ReLU()


    def get_name(self):
        return 'DeepFM_KLG'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        # process klg if config says so
        if self.klg_process == 'x_mlp':
            klg = self._process_klg_in_x_mlp(klg, x)

        embed_x = self.embedding(x)
        if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
            x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.input_dim)) + self.klg_predictor(klg)
        elif self.klg_usage_type == 'concat':
            mlp_input = torch.cat((embed_x.view(-1, self.input_dim), klg), dim=1)
            x = self.linear(x) + self.fm(embed_x) + self.mlp(mlp_input)
        return torch.sigmoid(x.squeeze(1))
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x
    
    def _process_klg_in_x_mlp(self, klg, x):
        # get params
        if self.share_emb:
            emb2_x = self.embedding(x).view(-1, self.x_num_fields * self.embed_dim)
        else:
            emb2_x = self.embedding2(x).view(-1, self.x_num_fields * self.embed_dim2)
        
        params = self.x_projector(emb2_x)

        # weights and bias
        weights_list = []
        bias_list = []
        for i in range(self.klg_processer_mlp_layers):
            weight_lhs = i*(self.klg_encoder_vec_dim+1)*self.klg_encoder_vec_dim
            weight_rhs = weight_lhs + self.klg_encoder_vec_dim*self.klg_encoder_vec_dim
            weights_list.append(params[:,weight_lhs:weight_rhs].reshape((-1,self.klg_encoder_vec_dim,self.klg_encoder_vec_dim)))

            bias_lhs = (i+1)*self.klg_encoder_vec_dim*self.klg_encoder_vec_dim + i*self.klg_encoder_vec_dim
            bias_rhs = bias_lhs + self.klg_encoder_vec_dim
            bias_list.append(torch.unsqueeze(params[:,bias_lhs:bias_rhs],dim=1))
        
        # forward klg
        klg = klg.reshape([-1, self.klg_num_fields, self.klg_encoder_vec_dim])
        klg_x_mlp = klg
        for i in range(self.klg_processer_mlp_layers):
            klg_x_mlp = self.f_a(torch.matmul(klg_x_mlp, weights_list[i]) + bias_list[i])
        return klg_x_mlp.view(-1, self.klg_vec_dim)

class DeepFM(Rec):
    def __init__(self, model_config, data_config):
        super(DeepFM, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.x_num_fields * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)

    def get_name(self):
        return 'DeepFM'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.input_dim))
        return torch.sigmoid(x.squeeze(1))
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

class DeepFM_WOEmb(Rec):
    def __init__(self, model_config, data_config):
        super(DeepFM_WOEmb, self).__init__(model_config, data_config)
        self.linear = Linear(self.vocabulary_size)
        self.fm = FactorizationLayer(reduce_sum=True)
        self.input_dim = self.x_num_fields * self.embed_dim
        self.mlp = MultiLayerPerceptron(self.input_dim, self.hidden_dims, self.dropout)

    def get_name(self):
        return 'DeepFM_WOEmb'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1) # in here x_user etc are all embeddings with shape [B, F, D]
        x = self.fm(x) + self.mlp(x.view(-1, self.input_dim)) # the linear part is gone
        return torch.sigmoid(x.squeeze(1))

class DIN_KLG(Rec):
    def __init__(self, model_config, data_config, klg_type):
        super(DIN_KLG, self).__init__(model_config, data_config)
        if klg_type in ['pos_ratio', 'inner_product']:
            self.klg_vec_dim = self.klg_num_fields * 1
        elif klg_type in ['encoder_vec']:
            self.klg_vec_dim = self.klg_num_fields * self.klg_encoder_vec_dim
        elif klg_type in ['user_seq_rep']:
            self.klg_vec_dim = self.hist_num_fields * self.embed_dim

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.input_dim = self.x_num_fields * self.embed_dim

        if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
            mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim
            self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        elif self.klg_usage_type == 'concat':
            mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim + self.klg_vec_dim
            self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        
        self.output_layer = torch.nn.Linear(self.hidden_dims[-1], 1)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim, self.hist_num_fields * self.embed_dim)
        
        if self.klg_usage_type == 'tower-lr':
            self.klg_predictor = torch.nn.Linear(self.klg_vec_dim, 1)
        elif self.klg_usage_type == 'tower-mlp':
            self.klg_predictor = MultiLayerPerceptron(self.klg_vec_dim, self.klg_mlp_hidden_dims, self.dropout)
        
        if self.klg_process == 'x_mlp':
            self.total_params_num = self.klg_processer_mlp_layers * (self.klg_encoder_vec_dim + 1) * self.klg_encoder_vec_dim
                
            if self.share_emb:
                self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim, self.total_params_num)
            else:
                self.embedding2 = FeaturesEmbedding(self.vocabulary_size, self.embed_dim2)
                self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim2, self.total_params_num)
            
            if self.x_mlp_act == 'tanh':
                self.f_a = torch.nn.Tanh()
            elif self.x_mlp_act == 'relu':
                self.f_a = torch.nn.ReLU()

    def get_name(self):
        return 'DIN_KLG'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        # process klg if config says so
        if self.klg_process == 'x_mlp':
            x = torch.cat((x_user, x_item, x_context), dim=1)
            klg = self._process_klg_in_x_mlp(klg, x)

        user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, atten_score = self.att(item_emb, user_hist, hist_len)
        context_emb = self.embedding(x_context).view(-1, self.context_num_fields * self.embed_dim)
        
        if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
            inp = torch.cat((user_emb, item_emb, context_emb, user_rep), dim=1)
            out = self.output_layer(self.mlp(inp)).squeeze(1)
            return torch.sigmoid(out + self.klg_predictor(klg).squeeze(1))
        elif self.klg_usage_type == 'concat':
            inp = torch.cat((user_emb, item_emb, context_emb, user_rep, klg), dim=1)
            out = self.output_layer(self.mlp(inp)).squeeze(1)
            return torch.sigmoid(out)
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x
    
    def _process_klg_in_x_mlp(self, klg, x):
        # get params
        if self.share_emb:
            emb2_x = self.embedding(x).view(-1, self.x_num_fields * self.embed_dim)
        else:
            emb2_x = self.embedding2(x).view(-1, self.x_num_fields * self.embed_dim2)
        
        params = self.x_projector(emb2_x)

        # weights and bias
        weights_list = []
        bias_list = []
        for i in range(self.klg_processer_mlp_layers):
            weight_lhs = i*(self.klg_encoder_vec_dim+1)*self.klg_encoder_vec_dim
            weight_rhs = weight_lhs + self.klg_encoder_vec_dim*self.klg_encoder_vec_dim
            weights_list.append(params[:,weight_lhs:weight_rhs].reshape((-1,self.klg_encoder_vec_dim,self.klg_encoder_vec_dim)))

            bias_lhs = (i+1)*self.klg_encoder_vec_dim*self.klg_encoder_vec_dim + i*self.klg_encoder_vec_dim
            bias_rhs = bias_lhs + self.klg_encoder_vec_dim
            bias_list.append(torch.unsqueeze(params[:,bias_lhs:bias_rhs],dim=1))
        
        # forward klg
        klg = klg.reshape([-1, self.klg_num_fields, self.klg_encoder_vec_dim])
        klg_x_mlp = klg
        for i in range(self.klg_processer_mlp_layers):
            klg_x_mlp = self.f_a(torch.matmul(klg_x_mlp, weights_list[i]) + bias_list[i])
        return klg_x_mlp.view(-1, self.klg_vec_dim)

# with a gate
class DIN_KLG_WG(Rec):
    def __init__(self, model_config, data_config, klg_type):
        super(DIN_KLG_WG, self).__init__(model_config, data_config)
        if klg_type in ['pos_ratio', 'inner_product']:
            self.klg_vec_dim = self.klg_num_fields * 1
        elif klg_type in ['encoder_vec']:
            self.klg_vec_dim = self.klg_num_fields * self.klg_encoder_vec_dim

        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        # if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
        mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        
        # elif self.klg_usage_type == 'concat':
        #     mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim + self.klg_vec_dim
        #     self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        
        self.output_layer = torch.nn.Linear(self.hidden_dims[-1], 1)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim, self.hist_num_fields * self.embed_dim)
        
        if self.klg_usage_type == 'tower-lr':
            self.klg_predictor = torch.nn.Linear(self.klg_vec_dim, 1)
        elif self.klg_usage_type == 'tower-mlp':
            self.klg_predictor = MultiLayerPerceptron(self.klg_vec_dim, self.klg_mlp_hidden_dims, self.dropout)

        self.gate = torch.nn.Parameter(torch.zeros(1,2))

    def get_name(self):
        return 'DIN_KLG_WG'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, atten_score = self.att(item_emb, user_hist, hist_len)
        context_emb = self.embedding(x_context).view(-1, self.context_num_fields * self.embed_dim)
        
        # if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
        inp = torch.cat((user_emb, item_emb, context_emb, user_rep), dim=1)
        out = self.output_layer(self.mlp(inp)).squeeze(1)
        
        alpha_vec = F.softmax(self.gate, dim=1)

        pred = alpha_vec[0,0] * torch.sigmoid(out) + alpha_vec[0,1] * torch.sigmoid(self.klg_predictor(klg).squeeze(1))
        return pred
        # return torch.sigmoid(out + self.klg_predictor(klg).squeeze(1))
        # elif self.klg_usage_type == 'concat':
        #     inp = torch.cat((user_emb, item_emb, context_emb, user_rep, klg), dim=1)
        #     out = self.output_layer(self.mlp(inp)).squeeze(1)
        #     return torch.sigmoid(out)
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

class DIN(Rec):
    def __init__(self, model_config, data_config):
        super(DIN, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        self.output_layer = torch.nn.Linear(self.hidden_dims[-1], 1)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim, self.hist_num_fields * self.embed_dim)
        
    def get_name(self):
        return 'DIN'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, atten_score = self.att(item_emb, user_hist, hist_len)
        context_emb = self.embedding(x_context).view(-1, self.context_num_fields * self.embed_dim)
        inp = torch.cat((user_emb, item_emb, context_emb, user_rep), dim=1)
        out = self.output_layer(self.mlp(inp)).squeeze(1)
        return torch.sigmoid(out)
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

class DIN_WOEmb(Rec):
    def __init__(self, model_config, data_config):
        super(DIN_WOEmb, self).__init__(model_config, data_config)
        mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        self.output_layer = torch.nn.Linear(self.hidden_dims[-1], 1)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim, self.hist_num_fields * self.embed_dim)
        
    def get_name(self):
        return 'DIN_WOEmb'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        user_emb = x_user.view(-1, self.user_num_fields * self.embed_dim)
        item_emb = x_item.view(-1, self.item_num_fields * self.embed_dim)
        user_hist = user_hist.view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, atten_score = self.att(item_emb, user_hist, hist_len)
        context_emb = x_context.view(-1, self.context_num_fields * self.embed_dim)
        inp = torch.cat((user_emb, item_emb, context_emb, user_rep), dim=1)
        out = self.output_layer(self.mlp(inp)).squeeze(1)
        return torch.sigmoid(out)

class DCNV2(Rec):
    def __init__(self, model_config, data_config):
        super(DCNV2, self).__init__(model_config, data_config)
        # load and compute parameters info
        self.mixed = model_config["mixed"]
        self.cross_layer_num = model_config["cross_layer_num"]
        self.embedding_size = model_config["embed_dim"]
        self.mlp_hidden_size = model_config["hidden_dims"]
        self.dropout_prob = model_config["dropout"]

        if self.mixed:
            self.expert_num = model_config["expert_num"]
            self.low_rank = model_config["low_rank"]

        self.in_feature_num = self.x_num_fields * self.embedding_size

        # define cross layers and bias
        if self.mixed:
            # U: (in_feature_num, low_rank)
            self.cross_layer_u = nn.ParameterList(
                nn.Parameter(
                    torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
                )
                for _ in range(self.cross_layer_num)
            )
            # V: (in_feature_num, low_rank)
            self.cross_layer_v = nn.ParameterList(
                nn.Parameter(
                    torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
                )
                for _ in range(self.cross_layer_num)
            )
            # C: (low_rank, low_rank)
            self.cross_layer_c = nn.ParameterList(
                nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
                for _ in range(self.cross_layer_num)
            )
            self.gating = nn.ModuleList(
                nn.Linear(self.in_feature_num, 1) for _ in range(self.expert_num)
            )
        else:
            # W: (in_feature_num, in_feature_num)
            self.cross_layer_w = nn.ParameterList(
                nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num))
                for _ in range(self.cross_layer_num)
            )
        # bias: (in_feature_num, 1)
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1))
            for _ in range(self.cross_layer_num)
        )

        # define deep and predict layers
        self.mlp_layers = MultiLayerPerceptron(self.in_feature_num, self.mlp_hidden_size, dropout=self.dropout_prob, output_layer=False)
        self.predict_layer = nn.Linear(self.in_feature_num + self.mlp_hidden_size[-1], 1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        

    def get_name(self):
        return 'DCNV2'

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.
        .. math:: x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`W_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of cross network, [batch_size, x_num_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l

        x_l = x_l.squeeze(dim=2)
        return x_l

    def cross_network_mix(self, x_0):
        r"""Cross network part of DCN-mix, which add MoE and nonlinear transformation in low-rank space.
        .. math::
            x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
        .. math::
            E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)
        :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
        :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
        :math:`g` is the nonlinear activation function.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of mixed cross network, [batch_size, x_num_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            expert_output_list = []
            gating_output_list = []
            for expert in range(self.expert_num):
                # compute gating output
                gating_output_list.append(
                    self.gating[expert](x_l.squeeze(dim=2))
                )  # (batch_size, 1)

                # project to low-rank subspace
                xl_v = torch.matmul(
                    self.cross_layer_v[i][expert].T, x_l
                )  # (batch_size, low_rank, 1)

                # nonlinear activation in subspace
                xl_c = self.tanh(xl_v)
                xl_c = torch.matmul(
                    self.cross_layer_c[i][expert], xl_c
                )  # (batch_size, low_rank, 1)
                xl_c = self.tanh(xl_c)

                # project back feature space
                xl_u = torch.matmul(
                    self.cross_layer_u[i][expert], xl_c
                )  # (batch_size, in_feature_num, 1)

                # dot with x_0
                xl_dot = xl_u + self.bias[i]
                xl_dot = torch.mul(x_0, xl_dot)

                expert_output_list.append(
                    xl_dot.squeeze(dim=2)
                )  # (batch_size, in_feature_num)

            expert_output = torch.stack(
                expert_output_list, dim=2
            )  # (batch_size, in_feature_num, expert_num)
            gating_output = torch.stack(
                gating_output_list, dim=1
            )  # (batch_size, expert_num, 1)
            moe_output = torch.matmul(
                expert_output, self.softmax(gating_output)
            )  # (batch_size, in_feature_num, 1)
            x_l = x_l + moe_output

        x_l = x_l.squeeze(dim=2)  # (batch_size, in_feature_num)
        return x_l

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        dcn_all_embeddings = self.embedding(x).view(-1, self.x_num_fields * self.embed_dim)  # (batch_size, x_num_field * embed_dim)
        deep_output = self.mlp_layers(
            dcn_all_embeddings
        )  # (batch_size, mlp_hidden_size)
        if self.mixed:
            cross_output = self.cross_network_mix(
                dcn_all_embeddings
            )  # (batch_size, in_feature_num)
        else:
            cross_output = self.cross_network(dcn_all_embeddings)
        concat_output = torch.cat(
            [cross_output, deep_output], dim=-1
        )  # (batch_size, in_num + mlp_size)
        output = self.sigmoid(self.predict_layer(concat_output))

        return output.squeeze(dim=1)

    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x


class DCNV2_KLG(Rec):
    def __init__(self, model_config, data_config, klg_type):
        super(DCNV2_KLG, self).__init__(model_config, data_config)
        '''klg part'''
        if klg_type in ['pos_ratio', 'inner_product']:
            self.klg_vec_dim = self.klg_num_fields * 1
        elif klg_type in ['encoder_vec']:
            self.klg_vec_dim = self.klg_num_fields * self.klg_encoder_vec_dim
        elif klg_type in ['user_seq_rep']:
            self.klg_vec_dim = self.hist_num_fields * self.embed_dim
        ''''''

        # load and compute parameters info
        self.mixed = model_config["mixed"]
        self.cross_layer_num = model_config["cross_layer_num"]
        self.embedding_size = model_config["embed_dim"]
        self.mlp_hidden_size = model_config["hidden_dims"]
        self.dropout_prob = model_config["dropout"]

        if self.mixed:
            self.expert_num = model_config["expert_num"]
            self.low_rank = model_config["low_rank"]

        self.in_feature_num = self.x_num_fields * self.embedding_size

        # define cross layers and bias
        if self.mixed:
            # U: (in_feature_num, low_rank)
            self.cross_layer_u = nn.ParameterList(
                nn.Parameter(
                    torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
                )
                for _ in range(self.cross_layer_num)
            )
            # V: (in_feature_num, low_rank)
            self.cross_layer_v = nn.ParameterList(
                nn.Parameter(
                    torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
                )
                for _ in range(self.cross_layer_num)
            )
            # C: (low_rank, low_rank)
            self.cross_layer_c = nn.ParameterList(
                nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
                for _ in range(self.cross_layer_num)
            )
            self.gating = nn.ModuleList(
                nn.Linear(self.in_feature_num, 1) for _ in range(self.expert_num)
            )
        else:
            # W: (in_feature_num, in_feature_num)
            self.cross_layer_w = nn.ParameterList(
                nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num))
                for _ in range(self.cross_layer_num)
            )
        # bias: (in_feature_num, 1)
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1))
            for _ in range(self.cross_layer_num)
        )

        '''klg part'''
        # define deep and predict layers
        if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
            self.mlp_layers = MultiLayerPerceptron(self.in_feature_num, self.mlp_hidden_size, dropout=self.dropout_prob, output_layer=False)
            self.predict_layer = nn.Linear(self.in_feature_num + self.mlp_hidden_size[-1], 1)
        elif self.klg_usage_type in ['concat']:
            self.mlp_layers = MultiLayerPerceptron(self.in_feature_num+self.klg_vec_dim, self.mlp_hidden_size, dropout=self.dropout_prob, output_layer=False)
            self.predict_layer = nn.Linear(self.in_feature_num + self.mlp_hidden_size[-1], 1)
        ''''''

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        
        '''klg part'''
        if self.klg_usage_type == 'tower-lr':
            self.klg_predictor = torch.nn.Linear(self.klg_vec_dim, 1)
        elif self.klg_usage_type == 'tower-mlp':
            self.klg_predictor = MultiLayerPerceptron(self.klg_vec_dim, self.klg_mlp_hidden_dims, self.dropout)
        
        if self.klg_process == 'x_mlp':
            self.total_params_num = self.klg_processer_mlp_layers * (self.klg_encoder_vec_dim + 1) * self.klg_encoder_vec_dim
                
            if self.share_emb:
                self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim, self.total_params_num)
            else:
                self.embedding2 = FeaturesEmbedding(self.vocabulary_size, self.embed_dim2)
                self.x_projector = torch.nn.Linear(self.x_num_fields*self.embed_dim2, self.total_params_num)
            
            if self.x_mlp_act == 'tanh':
                self.f_a = torch.nn.Tanh()
            elif self.x_mlp_act == 'relu':
                self.f_a = torch.nn.ReLU()
        ''''''

    def get_name(self):
        return 'DCNV2_KLG'

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.
        .. math:: x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`W_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of cross network, [batch_size, x_num_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l

        x_l = x_l.squeeze(dim=2)
        return x_l

    def cross_network_mix(self, x_0):
        r"""Cross network part of DCN-mix, which add MoE and nonlinear transformation in low-rank space.
        .. math::
            x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
        .. math::
            E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)
        :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
        :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
        :math:`g` is the nonlinear activation function.
        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.
        Returns:
            torch.Tensor:output of mixed cross network, [batch_size, x_num_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            expert_output_list = []
            gating_output_list = []
            for expert in range(self.expert_num):
                # compute gating output
                gating_output_list.append(
                    self.gating[expert](x_l.squeeze(dim=2))
                )  # (batch_size, 1)

                # project to low-rank subspace
                xl_v = torch.matmul(
                    self.cross_layer_v[i][expert].T, x_l
                )  # (batch_size, low_rank, 1)

                # nonlinear activation in subspace
                xl_c = self.tanh(xl_v)
                xl_c = torch.matmul(
                    self.cross_layer_c[i][expert], xl_c
                )  # (batch_size, low_rank, 1)
                xl_c = self.tanh(xl_c)

                # project back feature space
                xl_u = torch.matmul(
                    self.cross_layer_u[i][expert], xl_c
                )  # (batch_size, in_feature_num, 1)

                # dot with x_0
                xl_dot = xl_u + self.bias[i]
                xl_dot = torch.mul(x_0, xl_dot)

                expert_output_list.append(
                    xl_dot.squeeze(dim=2)
                )  # (batch_size, in_feature_num)

            expert_output = torch.stack(
                expert_output_list, dim=2
            )  # (batch_size, in_feature_num, expert_num)
            gating_output = torch.stack(
                gating_output_list, dim=1
            )  # (batch_size, expert_num, 1)
            moe_output = torch.matmul(
                expert_output, self.softmax(gating_output)
            )  # (batch_size, in_feature_num, 1)
            x_l = x_l + moe_output

        x_l = x_l.squeeze(dim=2)  # (batch_size, in_feature_num)
        return x_l

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x = torch.cat((x_user, x_item, x_context), dim=1)
        if self.klg_process == 'x_mlp':
            klg = self._process_klg_in_x_mlp(klg, x)

        dcn_all_embeddings = self.embedding(x).view(-1, self.x_num_fields * self.embed_dim)  # (batch_size, x_num_field * embed_dim)
        if self.klg_usage_type == 'concat':
            deep_output = self.mlp_layers(torch.cat((dcn_all_embeddings, klg),dim=1))  
        else:
            deep_output = self.mlp_layers(dcn_all_embeddings)
        if self.mixed:
            cross_output = self.cross_network_mix(
                dcn_all_embeddings
            )  # (batch_size, in_feature_num)
        else:
            cross_output = self.cross_network(dcn_all_embeddings)
        concat_output = torch.cat(
            [cross_output, deep_output], dim=-1
        )  # (batch_size, in_num + mlp_size)
        if self.klg_usage_type in ['tower-lr', 'tower-mlp']:
            output = self.sigmoid(self.predict_layer(concat_output)+self.klg_predictor(klg))
        elif self.klg_usage_type in ['concat']:
            output = self.sigmoid(self.predict_layer(concat_output))

        return output.squeeze(dim=1)
    
    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

    def _process_klg_in_x_mlp(self, klg, x):
        # get params
        if self.share_emb:
            emb2_x = self.embedding(x).view(-1, self.x_num_fields * self.embed_dim)
        else:
            emb2_x = self.embedding2(x).view(-1, self.x_num_fields * self.embed_dim2)
        params = self.x_projector(emb2_x)

        # weights and bias
        weights_list = []
        bias_list = []
        for i in range(self.klg_processer_mlp_layers):
            weight_lhs = i*(self.klg_encoder_vec_dim+1)*self.klg_encoder_vec_dim
            weight_rhs = weight_lhs + self.klg_encoder_vec_dim*self.klg_encoder_vec_dim
            weights_list.append(params[:,weight_lhs:weight_rhs].reshape((-1,self.klg_encoder_vec_dim,self.klg_encoder_vec_dim)))

            bias_lhs = (i+1)*self.klg_encoder_vec_dim*self.klg_encoder_vec_dim + i*self.klg_encoder_vec_dim
            bias_rhs = bias_lhs + self.klg_encoder_vec_dim
            bias_list.append(torch.unsqueeze(params[:,bias_lhs:bias_rhs],dim=1))
        
        # forward klg
        klg = klg.reshape([-1, self.klg_num_fields, self.klg_encoder_vec_dim])
        klg_x_mlp = klg
        for i in range(self.klg_processer_mlp_layers):
            klg_x_mlp = self.f_a(torch.matmul(klg_x_mlp, weights_list[i]) + bias_list[i])
        return klg_x_mlp.view(-1, self.klg_vec_dim)


class GRU4Rec(Rec):
    def __init__(self, model_config, data_config):
        super().__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        mlp_dim = (self.user_num_fields + self.item_num_fields + self.context_num_fields + self.hist_num_fields) * self.embed_dim
        self.mlp = MultiLayerPerceptron(mlp_dim, self.hidden_dims, self.dropout, output_layer=False)
        self.output_layer = torch.nn.Linear(self.hidden_dims[-1], 1)
        self.gru = torch.nn.GRU(self.hist_num_fields * self.embed_dim, self.hist_num_fields * self.embed_dim, batch_first=True)
    
    def get_name(self):
        return 'GRU4Rec'
    
    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        user_emb = self.embedding(x_user).view(-1, self.user_num_fields * self.embed_dim)
        item_emb = self.embedding(x_item).view(-1, self.item_num_fields * self.embed_dim)
        context_emb = self.embedding(x_context).view(-1, self.context_num_fields * self.embed_dim)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        hist_len = torch.where(hist_len > 0, hist_len, torch.ones_like(hist_len))
        
        hist_len_c = hist_len.cpu()
        user_hist_p = pack_padded_sequence(user_hist, hist_len_c, batch_first=True, enforce_sorted=False)
        hidden_seq, user_rep = self.gru(user_hist_p)
        # hidden_seq, user_rep = pad_packed_sequence(hidden_seq, batch_first=True)
        user_rep = torch.squeeze(user_rep, dim=0)

        inp = torch.cat((user_emb, item_emb, context_emb, user_rep), dim=1)
        out = self.output_layer(self.mlp(inp)).squeeze(1)
        return torch.sigmoid(out)
    
    def get_user_seq_rep(self, user_hist, hist_len):
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)
        hist_len = torch.where(hist_len > 0, hist_len, torch.ones_like(hist_len))
        
        hist_len_c = hist_len.cpu()
        user_hist_p = pack_padded_sequence(user_hist, hist_len_c, batch_first=True, enforce_sorted=False)
        hidden_seq, user_rep = self.gru(user_hist_p)
        user_rep = torch.squeeze(user_rep, dim=0)
        return user_rep

class AddEncoder(Rec):
    def __init__(self, model_config, data_config):
        super(AddEncoder, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.ui_cross = UICrossLayer()
        self.uic_cross = UICCrossLayer()

        if self.act == None:
            self.encoder = MultiLayerPerceptron(self.embed_dim * 2, self.hidden_dims + [self.klg_encoder_vec_dim], self.dropout, output_layer=False, batch_norm=False)
        else:
            self.encoder = MultiLayerPerceptron(self.embed_dim * 2, self.hidden_dims + [self.klg_encoder_vec_dim], self.dropout, output_layer=False, act=self.act, batch_norm=False)
        self.linear = torch.nn.Linear(self.klg_num_fields * self.klg_encoder_vec_dim, 1)

    def get_name(self):
        return 'AddEncoder'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x_user_emb = self.embedding(x_user)[:,self.klg_user_feats]
        x_item_emb = self.embedding(x_item)[:,self.klg_item_feats]
        x_context_emb = self.embedding(x_context)[:,self.klg_context_feats]

        x_cross = self.uic_cross(x_user_emb, x_item_emb, x_context_emb) # [B, F_pair3, 3D]
       
        x_klg = self.encoder(x_cross)
        x_klg = x_klg.reshape([-1, self.klg_num_fields * self.klg_encoder_vec_dim])
        pred = self.linear(x_klg)

        return torch.sigmoid(pred.squeeze(1))

    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

    def get_klg_vec(self, x_feat):
        embed_x = self.embedding(x_feat)
        input2encoder = embed_x.reshape([-1, self.embed_dim * 3])
        klg_vec = self.encoder(input2encoder) # [B, klg_encoder_vec_dim]
        return klg_vec

class AddEncoderWithHist(Rec):
    def __init__(self, model_config, data_config):
        super(AddEncoderWithHist, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        self.ui_cross = UICrossLayer()
        self.ih_cross = IHCrossLayer()
        self.uic_cross = UICCrossLayer()
        self.ihc_cross = IHCCrossLayer()

        
        self.linear = torch.nn.Linear(self.klg_num_fields * self.klg_encoder_vec_dim, 1)
        if self.act == None:
            self.encoder = MultiLayerPerceptron(self.embed_dim * 3, self.hidden_dims + [self.klg_encoder_vec_dim], self.dropout, output_layer=False, batch_norm=False)
        else:
            self.encoder = MultiLayerPerceptron(self.embed_dim * 3, self.hidden_dims + [self.klg_encoder_vec_dim], self.dropout, output_layer=False, act=self.act, batch_norm=False)

    def get_name(self):
        return 'AddEncoderWithHist'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        x_user_emb = self.embedding(x_user)[:,self.klg_user_feats]
        x_item_emb = self.embedding(x_item)[:,self.klg_item_feats]
        user_hist = self.embedding(user_hist)[:,:,self.klg_hist_feats]

        x_context_emb = self.embedding(x_context)[:,self.klg_context_feats]
        x_cross = self.uic_cross(x_user_emb, x_item_emb, x_context_emb) # [B, F_pair3, 3D]
        h_cross = self.ihc_cross(x_item_emb, user_hist, x_context_emb) # [B, T, F_pair3', 2D]
    

        x_klg = self.encoder(x_cross)
        x_klg = x_klg.reshape([-1, x_klg.shape[1] * x_klg.shape[2]])

        h_klg = self.encoder(h_cross) # [B, T, F_pair', Encoder_dim]
        mask = torch.arange(user_hist.shape[1])[None, :].to(hist_len.device) < hist_len[:, None]
        mask = torch.unsqueeze(torch.unsqueeze(mask.float(), 2), 3) # [B, T, 1, 1]

        h_klg = torch.sum(h_klg * mask, dim=1) # [B, F_pair', Encoder_dim]
        h_klg = h_klg.reshape([-1, h_klg.shape[1] * h_klg.shape[2]])

        x_klg = torch.cat((x_klg, h_klg), dim=1)
        pred = self.linear(x_klg)

        return torch.sigmoid(pred.squeeze(1))

    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

    def get_klg_vec(self, x_feat):
        embed_x = self.embedding(x_feat)
        input2encoder = embed_x.reshape([-1, self.embed_dim * 3])
        klg_vec = self.encoder(input2encoder) # [B, klg_encoder_vec_dim]
        return klg_vec

class Transformer(Rec):
    def __init__(self, model_config, data_config):
        super(Transformer, self).__init__(model_config, data_config)
        self.embedding = FeaturesEmbedding(self.vocabulary_size, self.embed_dim)
        klg_mlp_inp_dim = 3 * self.d_model
        if self.act == None:
            self.klg_mlp = MultiLayerPerceptron(klg_mlp_inp_dim, self.hidden_dims + [self.klg_encoder_vec_dim], self.dropout, output_layer=False, batch_norm=False)
        else:
            self.klg_mlp = MultiLayerPerceptron(klg_mlp_inp_dim, self.hidden_dims + [self.klg_encoder_vec_dim], self.dropout, output_layer=False, act=self.act, batch_norm=False)
        
        final_feature_triples = (self.user_num_fields + self.hist_num_fields) * self.item_num_fields * self.context_num_fields
        self.linear = torch.nn.Linear(final_feature_triples * self.klg_encoder_vec_dim, 1)
        self.encoder = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, batch_first=True)
        self.att = HistAttProd(self.item_num_fields * self.embed_dim, self.hist_num_fields * self.embed_dim)

        # generate pair index
        self.idx1, self.idx2, self.idx3 = [], [], []
        for i in range(self.user_num_fields + self.hist_num_fields):
            for j in range(self.user_num_fields + self.hist_num_fields, self.user_num_fields + self.hist_num_fields + self.item_num_fields):
                for k in range(self.user_num_fields + self.hist_num_fields + self.item_num_fields, self.user_num_fields + self.hist_num_fields + self.item_num_fields + self.context_num_fields):
                    self.idx1.append(i), self.idx2.append(j), self.idx3.append(k)

    def get_name(self):
        return 'Transformer'

    def forward(self, x_user, x_item, x_context, user_hist, hist_len, klg):
        user_emb = self.embedding(x_user)
        item_emb = self.embedding(x_item)
        context_emb = self.embedding(x_context)
        user_hist = self.embedding(user_hist).view(-1, user_hist.shape[1], user_hist.shape[2] * self.embed_dim)

        user_rep, _ = self.att(item_emb.view(-1, self.item_num_fields * self.embed_dim), user_hist, hist_len)
        user_rep = torch.reshape(user_rep, [-1, self.hist_num_fields, self.embed_dim])

        input2encoder = torch.cat((user_emb, user_rep, item_emb, context_emb), dim=1)
        encoder_out = self.encoder(input2encoder) # [B, F_num, d_model]

        encoder_out1 = encoder_out[:, self.idx1]
        encoder_out2 = encoder_out[:, self.idx2]
        encoder_out3 = encoder_out[:, self.idx3]
        
        klg_mlp_input = torch.cat((encoder_out1, encoder_out2, encoder_out3), dim=2)
        klg_mlp_out = self.klg_mlp(klg_mlp_input) # [B, F_num, encoder_vec_dim]
        klg_mlp_out = torch.reshape(klg_mlp_out, [-1, klg_mlp_out.shape[1] * klg_mlp_out.shape[2]])
        pred = self.linear(klg_mlp_out)
        
        return torch.sigmoid(pred.squeeze(1))

    def get_feat_embedding(self, x_feat):
        embed_x = self.embedding(x_feat)
        return embed_x

    def get_klg_vec(self, x_feat):
        embed_x = self.embedding(x_feat)
        klg_vec = self.encoder(embed_x).reshape([-1, self.d_model * 3]) # [B, 3 * d_model]
        klg_vec = self.klg_mlp(klg_vec)
        return klg_vec

