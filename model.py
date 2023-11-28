import gol
from dataloader import GraphData
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, dataset: GraphData):
        super(LightGCN, self).__init__()
        self.dataset = dataset
        self.n_user  = dataset.n_user
        self.n_item  = dataset.n_item

        self.hidden = gol.conf['hidden']
        self.n_layers = gol.conf['n_layers']
        self.keep_prob = gol.conf['keep_prob']
        self.embedding_user = nn.Embedding(self.n_user, self.hidden)
        self.embedding_item = nn.Embedding(self.n_item, self.hidden)

        if not gol.conf['load']:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            gol.pLog('use NORMAL distribution initilizer')
        else:
            # Not implemented
            gol.pLog('use pretarined data')

        self.Graph = self.dataset.getSparseGraph()

        gol.pLog(f"lgn is already to go (dropout:{gol.conf['dropout']})")
    
    def dropout(self, keep_prob):
        if (not self.training) or (not gol.conf['dropout']):
            return self.Graph
        size = self.Graph.size()
        index = self.Graph.indices().t()
        values = self.Graph.values()

        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob

        return torch.sparse_coo_tensor(index.t(), values, size)

    def propagate(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        g_droped = self.dropout(self.keep_prob)

        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        final_embs = torch.stack(embs, dim=1).mean(dim=1)

        users, items = torch.split(final_embs, [self.n_user, self.n_item])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def getLoss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = 0.5 * (userEmb0.norm(2).pow(2) +
                          posEmb0.norm(2).pow(2) +
                          negEmb0.norm(2).pow(2)) / users.size(0)

        pos_scores = (users_emb * pos_emb).sum(dim=-1)
        neg_scores = (users_emb * neg_emb).sum(dim=-1)
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()
        
        return bpr_loss, reg_loss
       
    def forward(self, users, items):
        all_users, all_items = self.propagate()
        user_emb = all_users[users]
        item_emb = all_items[items]
        return (user_emb * item_emb).sum(dim=-1)
