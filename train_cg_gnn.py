import pdb, random, sys
import torch
from torch.nn import Linear, Parameter
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Module
from torch.nn import Hardtanh
#from class_gnn import GCNConvClippedReLU
from scipy.interpolate import make_interp_spline
#from class_rational_gnn import GCNConvRational
#from simpleNNs import simpleNet
from IP import *
import copy
from sklearn.decomposition import PCA
from collections import Counter
import sklearn
from format_converters import *
from class_agg_comb_gnn_edge_attributes import *


################### Example to load data ######################
#edge_index = torch.tensor([[0, 1],
#                           [1, 0],
#                           [1, 2],
#                           [2, 1]], dtype=torch.long)
#x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#y = torch.tensor([[0], [1], [0]], dtype=torch.float)
#
#data = Data(x=x, y=y, edge_index=edge_index.t().contiguous())
#data_list = [data for x in range(100)]
#loader = DataLoader(data_list, batch_size=32, shuffle=True)
###############################################################


prefix = "/local_workspace/khalsamm"
N = int(sys.argv[1]) 
K = int(sys.argv[2])
nb_samples = int(sys.argv[3])
dataname = sys.argv[4]
cut_score = sys.argv[5]


#nb_samples = int(sys.argv[1])

sense = "maximize"
if dataname != "knapsack":
    sense = "minimize"

############################################## Load graphs ###########################################################################
######################################################################################################################################
loader_ = torch.load(f"{prefix}/{dataname}_{N}_{K}_{nb_samples}_graph_{cut_score}_samples.pkl", weights_only=False) 
# [instances_graphs, cuts, scores, indices_cuts_per_instance]
# instances_graphs is a list of [edge_index, x, edge_attr]
n_samples = len(loader_[0]) 

indices_cuts_per_instance = []
graphs_list = []
cuts_list_lhs = []
cuts_list_rhs = []
scores_list = []

#for i in range(n_samples):
#    print(i)
#    raw_data = [loader_[0][i], torch.tensor(loader_[1][i][0]), torch.tensor(loader_[1][i][1])]
#    raw_datas.append([raw_data, loader_[2][i]])
#
#imitation_datas = []
#cold_best_indices = []



for i in range(n_samples):
    try:
        graphs_list.append(loader_[0][i])
        cuts_list_lhs.append([xl[0] for xl in loader_[1][i]])
        cuts_list_rhs.append([xl[1] for xl in loader_[1][i]])
        scores_list.append(loader_[2][i])
    except:
        pdb.set_trace()




######################################################################################################################################
######################################################################################################################################
imitation_datas = []
cold_best_indices = []


#pdb.set_trace()

########################################################
############# IMITATION LEARNING #######################
########################################################
scores_I = []
i = -1
J = 0
nb_instance_to_nb_cuts = []
best_indices_track = []

def transformScore(scores):
    M_scores = max(scores)
    rscores = []
    for score in scores:
        if score == M_scores:
            rscores.append(1)
        else:
            rscores.append(0)
    return rscores


imitation_datas = []
score_datas = []
raw_data = []

# graphs_list
# scores_list

for i_instance, instance in enumerate(graphs_list):
    scores_instance = [sl for sl in scores_list[i_instance]] 
    imitation_datas_ = []
    max_score = np.max(scores_instance)
    ###################### Ignore instance if no useful cut ################
    if max_score <= 0:
        continue
    ########################################################################
    scores_I = transformScore(scores_instance)
    nb_above_zeroes = [x for x in scores_I if x > 0]
    index_max = np.argmax(scores_I)   
    cold_best_indices.append(index_max)
    index_min = np.argmin(scores_I)
    ##################### Loading cuts ####################################
    sum_scores = sum(scores_I)
    score_datas_ = [si/sum_scores for si in scores_I]
    imitation_datas_ = [(cll, clr) for cll, clr in zip(cuts_list_lhs[i_instance], cuts_list_rhs[i_instance])] 
    #imitation_datas_ = [imitation_datas_[index_min], imitation_datas_[index_max]]


    #score_datas_ = [si/sum_scores for si in scores_I]
    #imitation_datas_ = [raw_datas[indices[0] + index_cut] for index_cut in range(len(scores_I))] 
    #list_nb_cuts_per_instance_considered.append(len(indices))

    ################################## 
    ################################## 
    score_datas.append(score_datas_)  
    imitation_datas.append(imitation_datas_)
    raw_data.append((graphs_list[i_instance], imitation_datas_,score_datas_))

########################################################
########################################################
    
# raw_data: list of (graph instance, list cuts, list score)

########################################################
train_loader = raw_data[:int(0.65*nb_samples)]
val_loader =  raw_data[int(0.65*nb_samples):int(0.75*nb_samples)]
########################################################


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
##################################################### TRAINING #####################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
inner_dim_comb = int(sys.argv[6])
inner_outputs_dim = int(sys.argv[7])
#pdb.set_trace()
dim_in = len(train_loader[0][0]) - 1

final_out_dim = 1
dim_out = 10
dim_node_features = 3
dim_edge_features = 1
nb_iterations = int(sys.argv[8])

channels = [dim_node_features] + [inner_outputs_dim for i in range(nb_iterations-1)] + [final_out_dim]

#GNNModel = AggCombGNN(dim_node_features, dim_edge_features, nb_iterations=nb_iterations, inner_dim_comb = inner_dim_comb, final_out_dim = final_out_dim)
GNNModel = AggCombGNN(channels, dim_edge_features, nb_iterations=nb_iterations, inner_dim_comb = inner_dim_comb, final_out_dim = final_out_dim)
GNNModel = GNNModel.to(device)
GNNModel.train()

for name, param in GNNModel.named_parameters():    print(name, param.grad)
initial_learning_rate = 1e-2
weight_decay = 1e-2
#weight_decay = 1
#pdb.set_trace()
#params = list(GNNModel.parameters())
optimizer = torch.optim.AdamW(GNNModel.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)

#for name, param in model.named_parameters():    print(name, param.grad)

#num_epochs = 100
num_epochs = 50
batch_size = 10
loss_function = torch.nn.functional.cross_entropy
for epoch in range(num_epochs):
    #optimizer.zero_grad()
    if epoch > 0  and (epoch % 10 == 0): 
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*0.5

    #pdb.set_trace()
    indices_batches = [[batch_size*i + j for j in range(batch_size)] for i in range(int(len(train_loader)/batch_size))] 
    for ib in indices_batches:
        data_batch = [train_loader[bib] for bib in ib]
        cumulated_out = 0
        cumulated_out1 = 0

        optimizer.zero_grad()


        for data in data_batch:
            
            #edge_index = data[0][0][0]
            #x = data[0][0][1]
            #edge_attr = data[0][0][2]
    
            edge_index = data[0][0]
            x = data[0][1]
            edge_attr = data[0][2]

            #cuts = [sum([xcc.tolist() for xcc in xc], []) for xc in data[1]]
            ##### SOMETHING IS WRONG HERE  ABOVE, SHOULDNT DO THAT?
            #cuts = [sum([x.tolist(), y.tolist()], [])  for x,y in zip(data[1][0], data[1][1])]
            
            cuts = [sum([x[0].tolist(), x[1].tolist()], []) for x in data[1]]


            score_cuts = data[2]
            
            #score_cuts = [min(score_cuts), max(score_cuts)]


            
            nb_vars = len(cuts[0]) - 1 
            xfs = []
            eifs = []
            edfs = []
            cut_instance_embeddings = torch.zeros(len(cuts), requires_grad=True)
            ##############################################################################################
            ################################## GNN call on instance ####################################
            for icut, cut in enumerate(cuts):
                x_cut = [[xx, cu, cut[-1]] for xx, cu in zip(x[:nb_vars], cut[:-1])] + [[xx, xx, xx] for xx in x[nb_vars:]]
                edge_index_formatted = torch.tensor(edge_index, dtype=torch.int64).T 

                x_formatted = torch.tensor(x_cut, dtype=torch.float32)
                edge_attr_formatted = torch.tensor(edge_attr, dtype=torch.float32).reshape(len(edge_attr),1) 

                xfs.append(x_formatted)
                eifs.append(edge_index_formatted)
                edfs.append(edge_attr_formatted)


            #print(score_cuts)
            pdb.set_trace()

            try:
                cut_instance_embeddings = torch.hstack([torch.nn.functional.sigmoid(GNNModel(xf, eif, edf))[0][0] for xf, eif, edf in zip(xfs, eifs, edfs)])
            except:
                pdb.set_trace()

            cumulated_out += loss_function(cut_instance_embeddings, torch.tensor(score_cuts))

            ##############################################################################################
            ##############################################################################################



            #out = model(data[:, :dim_in])
            #pdb.set_trace()
        

        ########################################################################
        loss = cumulated_out
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        #pdb.set_trace()
        #print(GNNModel.combs[0].fc1.weight.grad) 
        #print(GNNModel.combs[1].fc1.weight.grad) 
        #print(GNNModel.combs[2].fc1.weight.grad) 
        #print(GNNModel.combs[3].fc1.weight.grad) 
        #print(GNNModel.combs[4].fc1.weight.grad) 



        #for name, param in model.named_parameters():    print(name, param.grad)
        #pdb.set_trace()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss validation: {loss_val.item():.4f}')
        ######################################################################
        ######################################################################


torch.save(GNNModel.state_dict(), f"{prefix}/GNNModel_{dataname}_{cut_score}_{nb_samples}_{dim_in}_{dim_out}_{num_epochs}_{batch_size}.torch_weights")





