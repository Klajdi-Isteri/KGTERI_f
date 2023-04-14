from abc import ABC
from operator import itemgetter

from torch_geometric.nn import LGConv
import torch
import torch_geometric
import numpy as np
import random


class LightGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 side,
                 n_layers,
                 adj,
                 random_seed,
                 name="LightGCN",
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
        self.n_layers = n_layers
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)
        self.alpha = torch.tensor([1 / (k + 1) for k in range(len(self.weight_size_list))])
        self.adj = adj

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # KGTERI Tensor
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

    def propagate_embeddings(self, evaluate=False):
        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [ego_embeddings]

        for layer in range(0, self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    all_embeddings += [list(
                        self.propagation_network.children()
                    )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]
            else:
                all_embeddings += [list(
                    self.propagation_network.children()
                )[layer](all_embeddings[layer].to(self.device), self.adj.to(self.device))]

        if evaluate:
            self.propagation_network.train()

        all_embeddings = sum([all_embeddings[k] * self.alpha[k] for k in range(len(all_embeddings))])
        gu, gi = torch.split(all_embeddings, [self.num_users, self.num_items], 0)

        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, -1)

        return xui

    def forward2(self, inputs, **kwargs):
        gu, gi, users, items = inputs
        users = users.reshape((users.shape[0],))
        items = items.reshape((items.shape[0],))
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
        gamma_i = gi.unsqueeze(0).expand(sparse.size(0), *gi.size()).to(self.device)
        # gamma_i_i tensore a tre dimensioni [u,i,emb]
        gamma_i_i = torch.sum(torch.bmm(sparse, gamma_i), 1).to(self.device)
        gamma_i_i = torch.nn.functional.normalize(gamma_i_i, p=2, dim=1)

        gamma_u = torch.squeeze(gu).to(self.device)
        xui = torch.sum(gamma_u * gamma_i_i, 1)

        return xui

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch

        # xu_pos = self.forward(inputs=(gu[user[:, 0]], gi[pos[:, 0]]))
        # xu_neg = self.forward(inputs=(gu[user[:, 0]], gi[neg[:, 0]]))

        # KGTERI forward
        xu_pos = self.forward2(inputs=(gu[user[:, 0]], gi[pos[:, 0]], user, pos))
        xu_neg = self.forward2(inputs=(gu[user[:, 0]], gi[neg[:, 0]], user, neg))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(self.Gu, 2) +
                               torch.norm(self.Gi, 2))
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)