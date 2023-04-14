"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC
from operator import itemgetter

import torch
import numpy as np
import random


class BPRMFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 side,
                 random_seed,
                 name="BPRMF",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w

        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.side = side

        x = []
        y = []
        z = []
        values = []

        for idx, csr_mat in enumerate(side):
            # converti la matrice CSR in un tensore sparso PyTorch
            coo_mat = csr_mat.tocoo()
            r = coo_mat.row.tolist()
            c = coo_mat.col.tolist()
            l = [idx] * len(r)
            x += l
            y += r
            z += c
            values += coo_mat.data.tolist()
        indices = torch.LongTensor([x, y, z])
        values = torch.FloatTensor(values)
        self.sparse_tensor = torch.sparse_coo_tensor(indices, values, (self.num_users, self.num_items, self.num_items))
        self.sparse_tensor = self.sparse_tensor.to(self.device)

    def forward(self, inputs, **kwargs):
        users, items = inputs
        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        gamma_i = torch.squeeze(self.Gi.weight[items]).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def forward2(self, inputs, **kwargs):
        users, items = inputs

        """
        First we perform the multiplication of the R and I, where R is the matrix containing 
        similarities and time information of each item (R's dimension is [n_items x n_items])
        and I is the embedding matrix of the items (I's dimension is [n_items x dim_embedding]).
        The result is a new I' matrix(aka. gamma_i_i) with new time and similarity enriched embeddings. 
        """
        sparse = torch.stack(tuple(itemgetter(*users)(self.sparse_tensor))).to(self.device)
        # prima permutazione per avere [i,i,u] e selezionare gli item scelti dalla posizione 0
        sparse = torch.stack(tuple(itemgetter(*items)(sparse.permute(1, 2, 0))))
        # seconda permutazione per avere [i,i,u] e selezionare gli item scelti dalla posizione 0
        sparse = torch.stack(tuple(itemgetter(*items)(sparse.permute(1, 0, 2))))
        # terza permutazione per tornare a [u,i,i] con gli specifici utenti e items dell'input
        sparse = sparse.permute(2, 0, 1)

        # gamma_i tensore a tre dimensioni [u,i,emb] e gamma_u tensore a tre dimensioni [u,i,i]
        gamma_i = self.Gi.weight[items].unsqueeze(0).expand(sparse.size(0), *self.Gi.weight[items].size()).to(self.device)
        # gamma_i_i tensore a tre dimensioni [u,i,emb]
        gamma_i_i = torch.sum(torch.bmm(sparse, gamma_i), 1).to(self.device)
        gamma_i_i = torch.nn.functional.normalize(gamma_i_i, p=2, dim=1)

        gamma_u = torch.squeeze(self.Gu.weight[users]).to(self.device)
        xui = torch.sum(gamma_u * gamma_i_i, 1)

        return xui, gamma_u, gamma_i_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu.weight[start:stop].to(self.device),
                            torch.transpose(self.Gi.weight.to(self.device), 0, 1))

    def train_step(self, batch):

        user, pos, neg = batch

        # xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(user[:, 0], pos[:, 0]))
        # xu_neg, _, gamma_i_neg = self.forward(inputs=(user[:, 0], neg[:, 0]))

        # KGTERI forward
        xu_pos, gamma_u, gamma_i_pos = self.forward2(inputs=(user[:, 0], pos[:, 0]))
        xu_neg, _, gamma_i_neg = self.forward2(inputs=(user[:, 0], neg[:, 0]))

        loss = -torch.mean(torch.nn.functional.logsigmoid(xu_pos - xu_neg))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
