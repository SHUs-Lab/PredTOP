import torch
from .models import GraphTransformer
import torch_geometric.utils as utils

import torch
from .train_utils import process_data, train_model, eval_model, get_device, pred_stats, process_graph_and_targets, fast_train_model, process_inference_graphs

from .gcn.model import GCN_simple, GAT_simple
from .gcn.utils import train_model as gcn_train_model
from .gcn.utils import eval_model as gcn_eval_model #, pred_stats
from .gcn.utils import fast_train_model as gcn_fast_train_model
from .gcn.utils import pred_stats as gcn_pred_stats
from .gcn.utils import process_inference_graphs as gcn_process_inference_graphs

MOD='transform'

def train(model_config, epochs=100, valid_epochs=50):
    train_fn = gcn_train_model if MOD == 'gcn' else train_model
    return train_fn(
        model_config['model'],
        model_config['train'],
        model_config['valid'],
        optimizer=model_config['optimizer'],
        criterion=model_config['criterion'],
        epochs=epochs, valid_epochs=valid_epochs,
        schedule=True
    )

def fast_train(model_config, epochs=100):
    train_fn = gcn_fast_train_model if MOD == 'gcn' else fast_train_model
    return train_fn(
        model_config['model'],
        model_config['train'],
        optimizer=model_config['optimizer'],
        criterion=model_config['criterion'],
        epochs=epochs,
        schedule=True
    )

def save(model_config):
    torch.save(model_config['model'], model_config['name'] + '.pth')

def eval(model_config, print_values=False, include_train=False, device=None):
    eval_fn = gcn_eval_model if MOD == 'gcn' else eval_model
    pred_fn = gcn_pred_stats if MOD == 'gcn' else pred_stats

    eval_fn(model_config['model'], model_config['valid'], print_values=print_values)
    pred_fn(model_config['model'], model_config['train'], model_config['valid'], include_train=include_train, device=device)

def make_model(
    train_loader, val_loader, deg,
    num_features = 46,
    num_classes = 1,
    nhid = 256,
    gps = 0,
    pe = 'dagpe',
    dropout_ratio=0.2,
    num_heads = 16,
    num_layers = 2,
    SAT = False,
    lr = 0.0001,
    weight_decay=1e-6,
    epochs = 1000,
    device=None,
    mod='transform'
):
    if device==None:
        device=get_device()

    if MOD == 'transform':
        model = GraphTransformer(in_size=num_features,
            num_class=num_classes,
            d_model=nhid,
            gps=gps,
            abs_pe=pe,
            dim_feedforward=4*nhid,
            dropout=dropout_ratio,
            num_heads=num_heads,
            num_layers=num_layers,
            batch_norm=True,
            in_embed=False,
            SAT=SAT,
            deg=deg,
            edge_embed=False,
            use_global_pool=True,    
            # add global pool
            global_pool='add'                   
        ).to(device)
    elif MOD == 'gcn':
        model = GCN_simple(num_layers, num_features, nhid, nhid, pooling='add').to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.27, 0.73]).to(device))
    criterion = torch.nn.L1Loss().to(device)

    model.train()

    name = f"model_{nhid}_{num_layers}_{num_heads}_{lr}_{epochs}"
    model_config = {
        'name': name,
        'model': model,
        'train': train_loader,
        'valid': val_loader,
        'optimizer': optimizer,
        'criterion': criterion,
    }
    
    return model_config

def create_model(device=None):
    if device==None:
      device=get_device()
    else:
      device=torch.device("cuda:" + str(device))
    
    bs=16
            
    epochs = 500

    nhid = 256 if MOD == 'gcn' else 64
    nlayers = 6 if MOD == 'gcn' else 4
    model_f = make_model(
        None, None, None,
        nhid=nhid, num_layers=nlayers,
        epochs=epochs,
        lr=0.001,
        device=device,
        mod=MOD
    )

    return model_f   

def run(graphs, targets, device=None):
    if device==None:
      device=get_device()
    else:
      device=torch.device("cuda:" + str(device))
    
    bs=32
    # DATAGEN: fix train split
    train_loader, val_loader = process_graph_and_targets(graphs, targets, scale_target=100, device=device, batch_size=bs, splits=[0.9, 0.1])
            
    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_loader])

    epochs = 500

    nhid = 256 if MOD == 'gcn' else 64
    nlayers = 6 if MOD == 'gcn' else 4
    model_f = make_model(
        train_loader, val_loader, deg,
        nhid=nhid, num_layers=nlayers,
        epochs=epochs,
        lr=0.001,
        device=device,
        mod=MOD
    )

    train(model_f, epochs)
    eval(model_f, False, device=device)

    return model_f

def pred(model, graphs, scale_target=100):
  device = get_device()
  outputs = torch.tensor([]).to(device)

  infr_fn = gcn_process_inference_graphs if MOD == 'gcn' else process_inference_graphs

  loader = infr_fn(graphs, device=device, batch_size=256)

  model.eval()

  if MOD == 'gcn':
    with torch.no_grad():
        for graph in loader: 
            out = model(graph.x, graph.edge_index, graph.batch)  # Perform a single forward pass.
            outputs = torch.cat((outputs, out), 0)
  elif MOD == 'transform':
    with torch.no_grad():
        for graph in loader: 
            out = model(graph)  # Perform a single forward pass.
            outputs = torch.cat((outputs, out), 0)

  res = (torch.squeeze(outputs).detach().cpu().numpy())/scale_target
  return res
