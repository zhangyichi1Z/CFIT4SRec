# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import math
import random

import numpy as np
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.fft as fft
import torch.nn.functional as F


class CFIT4SRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(CFIT4SRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.batch_size = config['train_batch_size']
        self.lmd = config['lmd']
        self.tau = config['tau']
        self.sim = config['sim']

        self.tau_plus = config['tau_plus']
        self.beta = config['beta']

        self.l_ok = config['l_ok']
        self.h_ok = config['h_ok']
        self.b_ok = config['b_ok']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )


        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if not config['low_r']:
            self.low_r = self.max_seq_length // 4
        else:
            self.low_r = config['low_r']
        if not config['high_r']:
            self.high_r = self.max_seq_length // 4
        else:
            self.high_r = config['high_r']

        self.LPA = self.createLPAilter((self.max_seq_length, self.hidden_size), self.low_r)
        self.HPA = self.createHPAilter((self.max_seq_length, self.hidden_size), self.high_r)

        self.BSA = [self.createBSAilter((self.max_seq_length, self.hidden_size), i, 2)
                    for i in range(min(self.max_seq_length, self.hidden_size) // 2 + 1)]

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        extended_attention_mask = self.get_attention_mask(item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        output_aug_l1, output_aug_l2, output_aug_h1, output_aug_h2, output_aug_b1, output_aug_b2 = [None for i in
                                                                                                    range(6)]

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)

        if self.l_ok:

            input_emb_aug_l1 = self.fft_2(input_emb, self.LPA)
            input_emb_aug_l2 = self.fft_2(input_emb, self.LPA)
            input_emb_aug_l1 = self.dropout(input_emb_aug_l1)
            input_emb_aug_l2 = self.dropout(input_emb_aug_l2)
            input_emb_aug_l1 = self.trm_encoder(input_emb_aug_l1, extended_attention_mask,
                                                output_all_encoded_layers=True)
            output_aug_l1 = input_emb_aug_l1[-1]

            input_emb_aug_l2 = self.trm_encoder(input_emb_aug_l2, extended_attention_mask,
                                                output_all_encoded_layers=True)
            output_aug_l2 = input_emb_aug_l2[-1]

        if self.h_ok:

            input_emb_aug_h1 = self.fft_2(input_emb, self.HPA)
            input_emb_aug_h2 = self.fft_2(input_emb, self.HPA)
            input_emb_aug_h1 = self.dropout(input_emb_aug_h1)
            input_emb_aug_h2 = self.dropout(input_emb_aug_h2)
            input_emb_aug_h1 = self.trm_encoder(input_emb_aug_h1, extended_attention_mask,
                                                output_all_encoded_layers=True)
            output_aug_h1 = input_emb_aug_h1[-1]

            input_emb_aug_h2 = self.trm_encoder(input_emb_aug_h2, extended_attention_mask,
                                                output_all_encoded_layers=True)
            output_aug_h2 = input_emb_aug_h2[-1]

        if self.b_ok:

            input_emb_aug_b1 = self.fft_2(input_emb, random.choice(self.BSA))
            input_emb_aug_b2 = self.fft_2(input_emb, random.choice(self.BSA))
            input_emb_aug_b1 = self.dropout(input_emb_aug_b1)
            input_emb_aug_b2 = self.dropout(input_emb_aug_b2)
            input_emb_aug_b1 = self.trm_encoder(input_emb_aug_b1, extended_attention_mask,
                                                output_all_encoded_layers=True)
            output_aug_b1 = input_emb_aug_b1[-1]

            input_emb_aug_b2 = self.trm_encoder(input_emb_aug_b2, extended_attention_mask,
                                                output_all_encoded_layers=True)
            output_aug_b2 = input_emb_aug_b2[-1]

        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        return output, output_aug_l1, output_aug_l2, output_aug_h1, output_aug_h2, output_aug_b1, output_aug_b2

    def my_fft(self, seq):
        f = torch.fft.rfft(seq, dim=1)
        amp = torch.absolute(f)
        phase = torch.angle(f)
        return amp, phase

    def fft_2(self, x, filter):
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift.cuda() * filter.cuda())))

    def createBSAilter(self, shape, bandCenter, bandWidth):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        bsFilter = torch.zeros((rows, cols))

        if min(rows, cols) // 2 == bandCenter:
            bsFilter[d < (bandCenter - bandWidth / 2)] = 1
        else:
            bsFilter[d > (bandCenter + bandWidth / 2)] = 1
            bsFilter[d < (bandCenter - bandWidth / 2)] = 1

        return bsFilter

    def createLPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        lpFilter = torch.ones((rows, cols))
        lpFilter[d > bandCenter] = 0

        return lpFilter

    def createHPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        hpFilter = torch.ones((rows, cols))
        hpFilter[d < bandCenter] = 0

        return hpFilter

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, seq_output_aug_l1, seq_output_aug_l2, \
        seq_output_aug_h1, seq_output_aug_h2, seq_output_aug_b1, seq_output_aug_b2 = self.forward(item_seq,
                                                                                                  item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'

            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # Constive loss
        nce_loss_l, nce_loss_h, nce_loss_b = [0, 0, 0]
        if self.l_ok:
            nce_loss_l = self.ncelosss(self.tau, 'cuda', seq_output_aug_l1, seq_output_aug_l2)

        if self.h_ok:
            nce_loss_h = self.ncelosss(self.tau, 'cuda', seq_output_aug_h1, seq_output_aug_h2)

        if self.b_ok:
            nce_loss_b = self.ncelosss(self.tau, 'cuda', seq_output_aug_b1, seq_output_aug_b2)

        return loss + self.lmd * nce_loss_l + self.lmd * nce_loss_h + self.lmd * nce_loss_b

    def ncelosss(self, temperature, device, batch_sample_one, batch_sample_two):
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        b_size = batch_sample_one.shape[0]
        batch_sample_one = batch_sample_one.view(b_size, -1)
        batch_sample_two = batch_sample_two.view(b_size, -1)

        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _, _, _, _, _, _ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _, _, _, _, _, _ = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

