import math
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.nn import init
import torch as th

class REConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', num_type=4, weight=True, bias=True, activation=None):
        super(REConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.weight_type = nn.Parameter(th.ones(num_type))
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, type_info):
        with graph.local_scope():
            aggregate_fn = lambda edges: {'m': edges.src['h']}
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5) if self._norm == 'both' else 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                feat = feat * th.reshape(norm, shp)
            
            feat = th.matmul(feat, self.weight)
            graph.srcdata['h'] = feat * self.weight_type[type_info].reshape(-1, 1)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            
            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5) if self._norm == 'both' else 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                rst = rst * th.reshape(norm, shp)
            if self.bias is not None:
                rst = rst + self.bias
            if self._activation is not None:
                rst = self._activation(rst)
            return rst

class HADE(nn.Module):
    """
    Heterogeneity-Based Distribution Encoder (HADE)
    """
    def __init__(self, num_types, output_dim, num_layers=1):
        super(HADE, self).__init__()
        layers = []
        curr_dim = num_types
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, output_dim))
            layers.append(nn.ReLU())
            curr_dim = output_dim
        self.mlp = nn.Sequential(*layers)
        self.to_prototypes = nn.Linear(curr_dim, num_types)
        self.final_proj = nn.Linear(num_types, output_dim)

    def forward(self, g, r):
        # 1. Estimating the affinity of potential heterologous prototypes
        z = self.mlp(r)
        p = F.softmax(self.to_prototypes(z), dim=-1)
        
        # 2. Normalized neighborhood propagation to obtain statistical summaries
        # HADE aggregates the probability distributions of potential prototypes rather than dense, continuous features
        with g.local_scope():
            g.ndata['p'] = p
            g.update_all(fn.copy_u('p', 'm'), fn.mean('m', 'n_dist'))
            n_dist = g.ndata['n_dist']
            
        # 3. Project back to dense representation space 
        return torch.relu(self.final_proj(n_dist))

class AGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, att_dropout=0.5, emb_dropout=0.5, temper=1.0, rl=False, rl_dim=4, beta=1):
        super(AGTLayer, self).__init__()
        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension
        self.head_dim = embeddings_dimension // nheads
        self.leaky = nn.LeakyReLU(0.01)
        self.temper = temper
        self.rl_dim = rl_dim
        self.beta = beta
        self.linear_l = nn.Linear(embeddings_dimension, self.head_dim * nheads, bias=False)
        self.linear_r = nn.Linear(embeddings_dimension, self.head_dim * nheads, bias=False)
        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)
        if rl:
            self.r_source = nn.Linear(rl_dim, rl_dim * nheads, bias=False)
            self.r_target = nn.Linear(rl_dim, rl_dim * nheads, bias=False)
        self.linear_final = nn.Linear(self.head_dim * nheads, embeddings_dimension, bias=False)
        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)
        self.LN = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, rh=None):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)
        if rh is not None:
            r_k = self.r_source(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).transpose(1,2)
            r_q = self.r_target(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).permute(0, 2, 3, 1)
            score_r = r_k @ r_q
            score = score + self.beta * score_r
        score = score / self.temper
        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)
        context = score @ fr
        h_sa = context.transpose(1,2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)
        h = self.LN(h + fh)
        return h

class HINAC(nn.Module):
    def __init__(self, g, num_class, input_dimensions, embeddings_dimension=256, num_layers=2, num_gnns=4, nheads=4, dropout=0.25, temper=1.0, num_type=4, beta=1, num_hade_layers=1):
        super(HINAC, self).__init__()
        self.g = g
        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_gnns = num_gnns
        self.nheads = nheads
        # Aligned projection corresponding to the LAAC-processed data
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension) for in_dim in input_dimensions])
        
        self.hade = HADE(num_type, embeddings_dimension, num_hade_layers)

        self.GCNLayers = nn.ModuleList()
        self.RELayers = nn.ModuleList()
        self.GTLayers = nn.ModuleList()
        
        for layer in range(self.num_gnns):
            self.GCNLayers.append(GraphConv(embeddings_dimension, embeddings_dimension, activation=F.relu))
            
            self.RELayers.append(REConv(embeddings_dimension, embeddings_dimension, activation=F.relu, num_type=num_type))
            
        for layer in range(self.num_layers):
            # Heterogeneity-Aware Graph Transformer 
            self.GTLayers.append(AGTLayer(embeddings_dimension, nheads, dropout, dropout, temper=temper, rl=True, rl_dim=embeddings_dimension, beta=beta))
            
        self.Drop = nn.Dropout(dropout)
        self.Prediction = nn.Linear(embeddings_dimension, num_class)

    def forward(self, features_list, seqs, type_emb, node_type, norm=False):
        # 1. Semantic completion feature  
        h = [fc(feature) for fc, feature in zip(self.fc_list, features_list)]
        gh = torch.cat(h, 0)
        
        # 2. Heterogeneous distribution encoding logic (HADE)
        r_initial = type_emb[node_type] # (N, num_type)
        r = self.hade(self.g, r_initial) # (N, embeddings_dimension)
        
        # 3. Dual parallel encoding pathways (MACE & HADE pathways)
        for layer in range(self.num_gnns):
            gh = self.GCNLayers[layer](self.g, gh)
            gh = self.Drop(gh)
            r = self.RELayers[layer](self.g, r, node_type)
        
        # 4. Transformer layer processing
        h_seq = gh[seqs]
        r_seq = r[seqs]
        for layer in range(self.num_layers):
            h_seq = self.GTLayers[layer](h_seq, rh=r_seq)
            
        output = self.Prediction(h_seq[:, 0, :])
        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True) + 1e-12)
        return output