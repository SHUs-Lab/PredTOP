import torch
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, GATv2Conv, GATConv
from torch_geometric.nn.norm.batch_norm import BatchNorm
import torch.nn.functional as F

class GCN_simple(torch.nn.Module):
    def __init__(self, gnn_layers, in_channel, hidden_channel, linear_hidden, pooling='add'):
        super(GCN_simple, self).__init__()
        
        self.convi = GCNConv(in_channel, hidden_channel)
        conv_layers = []
        bn_layers = []

        for _ in range(gnn_layers):
            conv_layers.append(GCNConv(hidden_channel, hidden_channel))
            bn_layers.append(BatchNorm(hidden_channel))

        self.conv_layers = ModuleList(conv_layers)
        self.bn_layers = ModuleList(bn_layers)

        self.fc1 = Linear(hidden_channel, linear_hidden)
        self.fc2 = Linear(linear_hidden, linear_hidden)

        self.bn1 = BatchNorm1d(linear_hidden)
        self.bn2 = BatchNorm1d(linear_hidden)

        self.output = Linear(linear_hidden, 1)

        self.pooling=pooling

    def forward(self, x, edge_index, batch):
        h = self.convi(x, edge_index)
        h = h.relu()

        for i, l in enumerate(self.conv_layers):
            h = l(h, edge_index)
            h = h.relu()
            h = self.bn_layers[i](h)

        # print(h.shape)
        if self.pooling == 'mean':
            h = global_mean_pool(h, batch)
        elif self.pooling == 'add':
            h = global_add_pool(h, batch)
        else:
            raise Exception("invalid pooling")

        # print(h.shape)
        h = self.fc1(h)
        h = h.relu()
        h = self.bn1(h)
        h = self.fc2(h)
        h = h.relu()
        h = self.bn2(h)

        h = self.output(h)
        # h = h.sigmoid()

        return h

class GAT_simple(torch.nn.Module):
    def __init__(self, gnn_layers, in_channel, hidden_channel, linear_hidden, heads=8):
        super(GAT_simple, self).__init__()
        self.gatin = GATConv(in_channel, hidden_channel, heads=heads)
        gat_layers = []
        for _ in range(gnn_layers):
            gat_layers.append(GATConv(hidden_channel*heads, hidden_channel, heads=heads, dropout=0.6))
            
        self.gatout = GATConv(hidden_channel*heads, hidden_channel, heads=1, concat=False, dropout=0.6)
        self.gat_layers = ModuleList(gat_layers)
        
        
        self.fc1 = Linear(hidden_channel, linear_hidden)
        self.fc2 = Linear(linear_hidden, linear_hidden)

        self.bn1 = BatchNorm1d(linear_hidden)
        self.bn2 = BatchNorm1d(linear_hidden)

        self.output = Linear(linear_hidden , 1)

    def forward(self, x, edge_index, batch):
        # print(batch.shape)       
        h = self.gatin(x, edge_index)
        h = F.dropout(h, p=0.6, training=self.training)
        h = F.elu(h)
        for l in self.gat_layers:
            h = l(h, edge_index)
            h = F.dropout(h, p=0.6, training=self.training)
            h = F.elu(h)

        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gatout(h, edge_index)
        h = F.elu(h)
        
        h = global_mean_pool(h, batch)
    
        h = self.fc1(h)
        h = h.relu()
        h = self.bn1(h)
        h = self.fc2(h)
        h = h.relu()
        h = self.bn2(h)

        h = self.output(h)
        
        return h