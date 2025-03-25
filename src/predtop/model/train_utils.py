import pickle
import torch
import networkx as nx
from torch_geometric.loader import DataLoader
import numpy as np
import sys
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
from torch_geometric.utils import (to_undirected)

import copy

def node_iter(G):
   return G.nodes

def node_dict(G):
    node_dict = G.nodes
    return node_dict

operators = [
 'add',
 'add_any',
 'argmax',
 'conv_general_dilated',
 'cos',
 'cumsum',
 'div',
 'dot_general',
 'eq',
 'erf',
 'exp',
 'gt',
 'integer_pow',
 'iota',
 'log',
 'lt',
 'max',
 'min',
 'mul',
 'neg',
 'pad',
 'pow',
 'reduce_max',
 'reduce_sum',
 'reduce_window_max',
 'rev',
 'rsqrt',
 'select_and_scatter_add',
 'select_n',
 'slice',
 'sqrt',
 'stop_gradient',
 'sub',
 'transpose',
]

dtypes = ['bool', 'float16', 'float32', 'int32']

node_type = [
 'invar',
 'outvar',
 # 'intermediate',
 'literal',
 'op_node',
]

def op_norm(val):
    return torch.div(val, torch.add(val, 1))

def set_op_attr_0(g, gn=0):
    node_attrs = {}
    for node in g:
        attrs = {}
        if 'label' not in g.nodes[node]:
            # print(node)
            for op in operators:
                attrs[op] = 0

            for dt in dtypes:
                attrs[dt] = 0

            attrs.update({
                'op_0': 1,
                'op_1': 0,
                'op_2': 0,
                'op_3': 0,
                'float32': 1,
            })
            node_attrs[node] = attrs
            continue

        # if 'label' not in g.nodes[node]:
        #   print(node)
        #   print(g.nodes[node])
        #   print(gn)

        for op in operators:
            if g.nodes[node]['label'] == op:
                attrs[op] = 1
            else:
                attrs[op] = 0
            
        for dt in dtypes:
            if g.nodes[node]['dtype'] == dt:
                attrs[dt] = 1
            else:
                attrs[dt] = 0

        if 'shape' not in g.nodes[node] or g.nodes[node]['shape'] is None:
            node_attrs[node] = attrs.update({
                'op_0': 0,
                'op_1': 0,
                'op_2': 0,
                'op_3': 0,
            })
        else: 
            for op in range(4):
                # if 'shape' not in g.nodes[node]:
                #     attrs['op_' + str(op)] = 0
                if len(g.nodes[node]['shape']) > op:
                    attrs['op_' + str(op)] = g.nodes[node]['shape'][op]
                else:
                    attrs['op_' + str(op)] = 0

        del g.nodes[node]['shape']
        del g.nodes[node]['label']
        del g.nodes[node]['dtype']
        del g.nodes[node]['type']
        node_attrs[node] = attrs

    try:
        nx.set_node_attributes(g, node_attrs)
    except:
        print(node_attrs['1.0'], g)
        raise Exception


def set_op_attr(g, gn=0):
    all_types = []
    node_attrs = {}
    for node in g:
        # print(node, g.nodes[node])

        attrs = {}
        if 'remat' not in g.nodes[node] or not g.nodes[node]['remat']:
            attrs['remat'] = 0
        else:
            attrs['remat'] = 1
            
        if 'remat' in g.nodes[node]:
            del g.nodes[node]['remat']

        if 'type' not in g.nodes[node]:
            for nt in node_type:
                attrs[nt] = 0
            attrs['literal'] = 1
        else:
            for nt in node_type:           
                if g.nodes[node]['type'] not in node_type:
                    raise Exception("Invalid Type found")
                    # attrs[nt] = 0
                if g.nodes[node]['type'] == nt:
                    attrs[nt] = 1
                else:
                    attrs[nt] = 0

        if 'label' not in g.nodes[node]:
            for op in operators:
                attrs[op] = 0

            for dt in dtypes:
                attrs[dt] = 0

            attrs.update({
                'op_0': 1,
                'op_1': 0,
                'op_2': 0,
                'op_3': 0,
                'float32': 1,
            })
            node_attrs[node] = attrs
            continue

        for op in operators:
            if g.nodes[node]['label'] == op:
                attrs[op] = 1
            else:
                attrs[op] = 0
            
        for dt in dtypes:
            if g.nodes[node]['dtype'] == dt:
                attrs[dt] = 1
            else:
                attrs[dt] = 0

        if 'shape' not in g.nodes[node] or g.nodes[node]['shape'] is None:
            node_attrs[node] = attrs.update({
                'op_0': 0,
                'op_1': 0,
                'op_2': 0,
                'op_3': 0,
            })
        else: 
            for op in range(4):
                # if 'shape' not in g.nodes[node]:
                #     attrs['op_' + str(op)] = 0
                if len(g.nodes[node]['shape']) > op:
                    attrs['op_' + str(op)] = g.nodes[node]['shape'][op]
                else:
                    attrs['op_' + str(op)] = 0

        del g.nodes[node]['shape']
        del g.nodes[node]['label']
        del g.nodes[node]['dtype']
        del g.nodes[node]['type']
        node_attrs[node] = attrs

    try:
      nx.set_node_attributes(g, node_attrs)
    except:
      print(node_attrs['1.0'], g)
      raise Exception

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_intermediate(g):
    remove_nodes = []
    add_edges = []
    
    for n in g.nodes:
        if 'type' not in g.nodes[n]:
            continue
            
        if g.nodes[n]['type'] == 'intermediate':
            succ = list(g.successors(n))
            pred = list(g.predecessors(n))
            
            if len(pred) != 1:
                raise Exception('invalid!!')
                
            remove_nodes.append(n)
            for s in succ:
                add_edges.append((pred[0], s))

    for node in remove_nodes:
        g.remove_node(node)
        
    for edge in add_edges:
        g.add_edge(edge[0], edge[1])
        
    return g

def process_inference_graphs(graphs, device=None, batch_size=32):
  if device is None:
    device=get_device()
  
  del_nodes = []
  for i, g in enumerate(graphs):
    if len(g.nodes) <= 2:
        del_nodes.append(i)

  new_graphs = [val for idx, val in enumerate(graphs) if idx not in del_nodes]

  graphs = new_graphs
  for i, g in enumerate(graphs):
    g.remove_nodes_from(list(nx.isolates(g)))
    remove_intermediate(g)
    set_op_attr(g, i)

  type_values = operators + dtypes  + node_type # + ['remat']
  type_ops = ['op_0', 'op_1', 'op_2', 'op_3']  

  #convert networkx graphs to pyg graphs
  pyg_datas = []

  for i, graph in enumerate(graphs):
      g = graph
      data = from_networkx(g)
      data_x = torch.stack([data[t] for t in type_values] + [op_norm(data[t]) for t in type_ops])

      for tv in type_values:
          del data[tv]
      data.x = torch.transpose(data_x.float(), 0, 1)
      data = data.to(device)                                  
      
      graph_size = data.num_nodes
      pe = torch.from_numpy(np.array(range(graph_size))).long()
      data.abs_pe = pe
      data.dag_rr_edge_index = to_undirected(data.edge_index)
      pyg_datas.append(data)

  data_size = len(pyg_datas)
  print("Data Size: ", data_size)
  print('Batch Size: ', batch_size)
  return DataLoader(pyg_datas, batch_size=batch_size, shuffle=False)


def process_graph_and_targets(graphs, targets, scale_target=1, device=None, batch_size=32, splits=[0.8, 0.2], shuffle=True):
  if device is None:
     device = get_device()
    
  del_nodes = []
  for i, g in enumerate(graphs):
    if len(g.nodes) <= 2 or targets[i] == np.inf:
        del_nodes.append(i)

  new_graphs = [val for idx, val in enumerate(graphs) if idx not in del_nodes]
  new_targets = [val for idx, val in enumerate(targets) if idx not in del_nodes]

  graphs = new_graphs
  targets = new_targets
  for i, g in enumerate(graphs):
      g.remove_nodes_from(list(nx.isolates(g)))
      remove_intermediate(g)
      set_op_attr(g, i)

  targets = torch.tensor(targets, dtype=torch.float32) * scale_target
  type_values = operators + dtypes  + node_type # + ['remat']
  type_ops = ['op_0', 'op_1', 'op_2', 'op_3']

  #convert networkx graphs to pyg graphs
  pyg_datas = []
  for i, graph in enumerate(graphs):
      g = graph
      data = from_networkx(g)

      data_x = torch.stack([data[t] for t in type_values] + [op_norm(data[t]) for t in type_ops])

      for tv in type_values:
          del data[tv]
      data.x = torch.transpose(data_x.float(), 0, 1)
      data.y = torch.tensor([targets[i]], dtype=torch.float32)
      data = data.to(device)
                                  
      
      graph_size = data.num_nodes
      pe = torch.from_numpy(np.array(range(graph_size))).long()
      data.abs_pe = pe
        
      data.dag_rr_edge_index = to_undirected(data.edge_index)
    
      pyg_datas.append(data)

  data_size = len(pyg_datas)
  print("Data Size: ", data_size)

  if sum(splits) != 1:
        raise Exception("Invalid splits")

  split_sizes = [int(i * data_size) for i in splits [1:]]
  split_sizes = [data_size - sum(split_sizes)] + split_sizes
  print("Splits : ", split_sizes)

  split_datasets = torch.utils.data.random_split(pyg_datas, split_sizes)
  print('shuffle: ', shuffle)
  train_loader = DataLoader(split_datasets[0], batch_size=batch_size, shuffle=shuffle)

  remaining_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in split_datasets[1:]]

  return tuple([train_loader] + remaining_loaders)

def process_data(filename, graph_key, target_key, scale_target=1, device=None, batch_size=32, splits=[0.8, 0.2]):
  if type(graph_key) == str:
    graph_key = [graph_key]
    target_key = [target_key]

  graphs = []
  targets = []
  with open(filename, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    for i, g in enumerate(graph_key):
      graphs = graphs + data[g]
      targets = targets + data[target_key[i]]

  return process_graph_and_targets(graphs, targets, scale_target=scale_target, device=device, batch_size=batch_size, splits=splits)
#   graphs = graphs[:1]


def process_data_og(filename, graph_key, target_key, scale_target=1, device=None, batch_size=32):
  if device is None:
     device = get_device()

  if type(graph_key) == str:
    graph_key = [graph_key]
    target_key = [target_key]

  graphs = []
  targets = []
  with open(filename, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    for i, g in enumerate(graph_key):
      graphs = graphs + data[g]
      targets = targets + data[target_key[i]]

  for i, g in enumerate(graphs):
    if len(g.nodes) <= 2:
        del graphs[i]
        del targets[i]

  for i, g in enumerate(graphs):
      set_op_attr(g, i)

  targets = torch.tensor(targets, dtype=torch.float32) * scale_target
  type_values = operators + dtypes + ['op_0', 'op_1', 'op_2', 'op_3']

  #convert networkx graphs to pyg graphs
  pyg_datas = []
  
#   stdm = 0
#   stds = 1
#   for i, graph in enumerate(graphs):

  print('a')
  for i, graph in enumerate(graphs):
      g = graph
      data = from_networkx(g)
      # print(g)
      data_x = torch.stack([data[t] for t in type_values])

      for tv in type_values:
          del data[tv]
      data.x = torch.transpose(data_x.float(), 0, 1)
      # data.y = torch.log10(torch.tensor([targets[i]]))
      data.y = torch.tensor([targets[i]], dtype=torch.float32)
      data = data.to(device)
      pyg_datas.append(data)

  data_size = len(pyg_datas)
  train_size = int(0.8 * data_size)
  valid_size = data_size - train_size
  print("Data Size: ", data_size)
  print("Train Size: ", train_size)
  print("Valid Size: ", valid_size)

  train_dataset, valid_dataset = torch.utils.data.random_split(pyg_datas, [train_size, valid_size])
  # train_dataset.to(device)
  # valid_dataset.to(device)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset,  batch_size=batch_size, shuffle=False)

  return train_loader, valid_loader

def fast_train_model(model, train_loader, epochs=100, lr=0.001, optimizer=None, criterion=None, quiet=False, schedule=True):
  if optimizer is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
  
  if criterion is None:
    criterion = torch.nn.MSELoss()  # MSELoss function

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

  model.train()
  lbar = tqdm(range(epochs), ascii=True, file=sys.stdout) if not quiet else range(epochs)
  for epoch in lbar:
    for graph in train_loader: 
      optimizer.zero_grad()  # Clear gradients.

      out = model(graph) #, graph.edge_index, graph.batch)  # Perform a single forward pass.
      # print(torch.squeeze(out), graph.y)
      loss = criterion(torch.squeeze(out), graph.y)  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.

      # print(loss, '\t\t', train_loss, '\t\t', len(graph.x))
      # train_loss += loss.item()
      # train_count += 1
      optimizer.step()  # Update parameters based on gradients.

    # if (True or (epoch + 1) % valid_epochs == 0 or epoch < 10) and not quiet:
    #   model.eval()
    #   valid_loss = 0
    #   count = 0
    #   for graph in valid_loader:  
    #     optimizer.zero_grad()  # Clear gradients.
    #     # print('valid', '\t', loss, len(graph.x))
    #     out = model(graph) #, graph.edge_index, graph.batch)  # Perform a single forward pass.
    #     loss = criterion(torch.squeeze(out), graph.y)  # Compute the loss solely based on the training nodes.
    #     valid_loss += loss.item()
    #     count += 1

      # if ((epoch + 1) % valid_epochs == 0 or epoch < 10):
      #     print(f'Epoch: {epoch},\t Training Loss: {train_loss/train_count:.5f} \tValid Loss: {valid_loss/count:.5f} \t\tLr: {optimizer.param_groups[0]["lr"]:.10f}')
    scheduler.step()
  # return lrs, training_losses, valid_losses


def train_model(model, train_loader, valid_loader, epochs=100, lr=0.001, optimizer=None, criterion=None, valid_epochs=10, quiet=False, schedule=True):
  if criterion is None:
    criterion = torch.nn.MSELoss()  # MSELoss function

  if optimizer is None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

  lrs = []
  training_losses = []
  valid_losses = []
  
  best_loss = float('inf')
  best_model_weights = None
  pval = 200
  patience = pval

  lbar = tqdm(range(epochs), ascii=True, file=sys.stdout) if not quiet else range(epochs)
  for epoch in lbar:
    model.train()
    train_loss = 0
    train_count = 0
    for graph in train_loader: 
      optimizer.zero_grad()  # Clear gradients.

      out = model(graph) #, graph.edge_index, graph.batch)  # Perform a single forward pass.
      # print(torch.squeeze(out), graph.y)
      loss = criterion(torch.squeeze(out), graph.y)  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.

      # print(loss, '\t\t', train_loss, '\t\t', len(graph.x))
      train_loss += loss.item()
      train_count += 1
      optimizer.step()  # Update parameters based on gradients.

    if (True or (epoch + 1) % valid_epochs == 0 or epoch < 10) and not quiet:
      model.eval()
      valid_loss = 0
      count = 0
      for graph in valid_loader:  
        optimizer.zero_grad()  # Clear gradients.
        # print('valid', '\t', loss, len(graph.x))
        out = model(graph) #, graph.edge_index, graph.batch)  # Perform a single forward pass.
        loss = criterion(torch.squeeze(out), graph.y)  # Compute the loss solely based on the training nodes.
        valid_loss += loss.item()
        count += 1

      if ((epoch + 1) % valid_epochs == 0 or epoch < 10):
          tqdm.write(f'Epoch: {epoch},\t Training Loss: {train_loss/train_count:.5f} \tValid Loss: {valid_loss/count:.5f} \t\tLr: {optimizer.param_groups[0]["lr"]:.10f}')
      
      val_loss = valid_loss/count
      valid_losses.append(val_loss)
      training_losses.append(train_loss/train_count)

    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
        patience = pval  # Reset patience counter
    else:
        if patience > 0:
            patience -= 1
        if epoch > 0.2 * epochs:
            if patience == 0:
                break
  model.load_state_dict(best_model_weights)   
  return lrs, training_losses, valid_losses

def eval_model(model, valid_loader, print_values=False, acc_fn=0):
  def accuracy(pred, true):
    return ((pred - true)/true) * 100

  def rpd_acc(pred, true):
    return ((pred - true)/(torch.abs(pred) + torch.abs(true)) ) * 100

  count = 0
  total = 0

  model.eval()

  with torch.no_grad():
    for graph in valid_loader:
      out = model(graph)  # Perform a single forward pass.

      if acc_fn == 0:
          acc = accuracy(torch.squeeze(out), graph.y)
      else:
          acc = rpd_acc(torch.squeeze(out), graph.y)

      # print("Batch Error ", acc.abs().mean().item(), " %")
      # print()

      if print_values:
        print("Predicted\tTrue")
        for i, e in enumerate(out):
          print("%.4f\t\t%.4f\t\t%.4f" % (e.item(), graph.y[i].item(), acc[i]))
      total += acc.abs().mean().item()
      count += 1

  # print(total)
  # print(count)
  print("Final Error: ", total/count)

  return total/count

def pred_stats(model, train_loader, valid_loader, device=None, include_train=False):
  if device is None:
    device = get_device()
  outputs = torch.tensor([]).to(device)

  model.eval()
  with torch.no_grad():
    for graph in valid_loader: 
        out = model(graph)  # Perform a single forward pass.
        outputs = torch.cat((outputs, out), 0)

    if include_train:
      for graph in train_loader: 
          out = model(graph.x, graph.edge_index, graph.batch)  # Perform a single forward pass.
          outputs = torch.cat((outputs, out), 0)
  op = torch.squeeze(outputs)
  mean = torch.mean(op)
  std = torch.std(op)

  print("Prediction Stats: ")
  print("Mean: ", mean.item())
  print("Std: ", std.item())
  print("Std/Mean: ", (std/mean).item() )


def pred(model, graphs):
  device = get_device()
  rarr = [0] * len(graphs)
  outputs = torch.tensor([]).to(device)

  loader = process_graph_and_targets(graphs, rarr, scale_target=100, device=device, batch_size=512, shuffle=False)

  model.eval()
  with torch.no_grad():
    for graph in loader: 
        out = model(graph)  # Perform a single forward pass.
        outputs = torch.cat((outputs, out), 0)

  return outputs.detach().cpu().numpy()