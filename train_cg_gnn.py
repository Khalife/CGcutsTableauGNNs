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
from class_gnn import GCNConvClippedReLU
from scipy.interpolate import make_interp_spline
#from class_rational_gnn import GCNConvRational
from simpleNNs import simpleNet
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


prefix = "/local_workspace/khalsamm/"
N = int(sys.argv[1]) 
K = int(sys.argv[2])
nb_samples = int(sys.argv[3])


dataname = sys.argv[4]
#nb_samples = int(sys.argv[1])

sense = "maximize"
if dataname != "knapsack":
    sense = "minimize"

############################################## Load graphs ###########################################################################
######################################################################################################################################
loader_ = torch.load(f"{prefix}/data/{dataname}_{N}_{K}_{nb_samples}_graph_samples.pkl") 
# [instances_graphs, cuts, scores, indices_cuts_per_instance]
# instances_graphs is a list of [edge_index, x, edge_attr]
n_samples = len(loader_[0]) 

indices_cuts_per_instance = []
graphs_list = []
cuts_list_lhs = []
cuts_list_rhs = []
scores_list = []

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inner_dim_comb = int(sys.argv[5])
inner_outputs_dim = int(sys.argv[6])
#pdb.set_trace()
dim_in = len(train_loader[0][0]) - 1

final_out_dim = 1
dim_out = 10
dim_node_features = 3
dim_edge_features = 1
nb_iterations = int(sys.argv[7])

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
            
            edge_index = data[0][0][0]
            x = data[0][0][1]
            edge_attr = data[0][0][2]

            cuts = [sum([xcc.tolist() for xcc in xc], []) for xc in data[1]]

            score_cuts = data[2]


            
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


            cut_instance_embeddings = torch.hstack([torch.nn.functional.sigmoid(GNNModel(xf, eif, edf))[0][0] for xf, eif, edf in zip(xfs, eifs, edfs)])
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


torch.save(GNNModel.state_dict(), f"{prefix}/khalsamm/GNNModel_{dataname}_{nb_samples}_{dim_in}_{dim_out}_{num_epochs}_{batch_size}.torch_weights")

GNNModel.eval()
for param in GNNModel.parameters():
    param.requires_grad = False


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

########################################## Test ###############################################
#n_sample_instances = len(ip_instances)

n_sample_instances = int(0.75*len(raw_data))

test_loader = raw_data[n_sample_instances:]


parallelism_tree_sizes = []
random_tree_sizes = []
best_nn_tree_sizes = []
optimal_tree_sizes = []

#pdb.set_trace()
#scores_collect_test = scores_collect[int(0.75*(nb_samples)*nb_cuts_per_instance):]

# Select cut based on heuristic and NN
collection_test_cut_vectors = []
best_indices_nn = []
best_indices_parallelism = []
best_cut_indices_debug = []
delta_best_tree_sizes = []
distribution_scores = []
distribution_scores_para = []
optimal_tree_size_lols = []
optimal_value_lols = []
optimal_sol_lols = []
scores_cuts_debug = []

################### Lists used in the next loop ####################
#ip_instances_test
#ip_instances_cuts
#indices_cuts_per_instance  # test part
#raw_ip_instance_test


######################################################

for i, instance in enumerate(test_loader):
    # raw_data.append((graphs_list[i_instance], imitation_datas_,score_datas_))
    graph_i = instance[0] 
    cuts_i = instance[1]
    scores_i = instance[2]

    ################# Debug version ##############
    if i > 250:
        break
    ##############################################

    print(len(test_loader)-i)
    #A = copy.deepcopy(instance.A)
    #b = copy.deepcopy(instance.b)
    #c = copy.deepcopy(instance.c)
    #instance.optimize()
    #tree_size_before_cut = instance.treesize

    ####################################
    # Level zero: Best strategy
    gom_cuts = [[ci[0] for ci in cuts_i], [ci[1] for ci in cuts_i]]
    #pdb.set_trace()
    
    #######################################################################
    #######################################################################
    #######################################################################
    scores_debug_lol = []
    optimal_tree_size_lol = []
    instance_ = graphToInstance(graph_i, sense=sense)
    instance_.optimize() 
    tree_size_before_cut = instance_.treesize
    
    for j in range(len(gom_cuts[0])):
        lhs = gom_cuts[0][j]
        rhs = gom_cuts[1][j]
        instance_ = graphToInstance(graph_i, sense=sense)
        instance_.add_cut(lhs, rhs)

        instance_.optimize()
        optimal_sol_lols.append(instance_.x_IP)
        optimal_value_lols.append(np.dot(instance_.c, instance_.x_IP))
        optimal_tree_size_j = instance_.treesize
        optimal_tree_size_lol.append(optimal_tree_size_j)
        scores_debug_lol.append((tree_size_before_cut - optimal_tree_size_j)/tree_size_before_cut)
    
    ##############################################################################################
    ##################### Ignore instances where any cut is useless ##############################
    #if max(scores_debug_lol) <= 0:
    #    continue
    ##############################################################################################
    ##############################################################################################

    best_cut_index = np.argmax(scores_debug_lol)
    best_cut_indices_debug.append(best_cut_index)

    optimal_tree_size = optimal_tree_size_lol[best_cut_index] 
    delta_best_tree_sizes.append(optimal_tree_size - tree_size_before_cut)
    optimal_tree_sizes.append(optimal_tree_size)
    optimal_tree_size_lols.append(optimal_tree_size_lol)
    distribution_scores.append(scores_debug_lol) 
    #######################################################################
    #######################################################################
    #######################################################################


    
    ####################################
    ####################################
    # Heuristic 1: parallelism
    objective_vector = instance_.c
    scores_cuts_i = []
    for j in range(len(gom_cuts[0])):
        lhs_j = gom_cuts[0][j]
        scores_cuts_i.append(np.dot(objective_vector, lhs_j)/(np.linalg.norm(objective_vector)*np.linalg.norm(lhs_j)))
    
    if sense == "maximize": 
        best_cut_index_parallelism = np.argmax(scores_cuts_i)
    if sense == "minimize":
        best_cut_index_parallelism = np.argmax([-sci for sci in scores_cuts_i])

    best_lhs = gom_cuts[0][best_cut_index_parallelism]
    best_rhs = gom_cuts[1][best_cut_index_parallelism]

    distribution_scores_para.append(scores_cuts_i) 

    best_indices_parallelism.append(best_cut_index_parallelism)
    #x = raw_ip_instance_test[i]
    #instance_ = vector_to_ip(x[0], x[2], x[1], sense=sense)
    instance_ = graphToInstance(graph_i, sense=sense)
    instance_.add_cut(best_lhs, best_rhs)
    instance_.optimize()
    parallelism_tree_size = instance_.treesize 
    parallelism_tree_sizes.append(parallelism_tree_size)

    if parallelism_tree_size < optimal_tree_size:
        print(instance_.x_IP)
        print(np.dot(instance_.c, instance_.x_IP))
        print(optimal_sol_lols[-1])
        print(optimal_value_lols[-1])
    ####################################
    # Heuristic 2: random 
    random_cut_index = np.random.randint(len(gom_cuts[0]))
    random_lhs = gom_cuts[0][random_cut_index] 
    random_rhs = gom_cuts[1][random_cut_index] 

    #x = raw_ip_instance_test[i]
    #instance_ = vector_to_ip(x[0], x[2], x[1], sense=sense)
    instance_ = graphToInstance(graph_i, sense=sense)
    instance_.add_cut(random_lhs, random_rhs)  
    instance_.optimize()
    random_tree_size = instance_.treesize 
    random_tree_sizes.append(random_tree_size)
    ####################################
    # Heuristic 3: NN

    collection_test_cut_vector = []
    scores_nn = []
    #instance_i = raw_ip_instance_test[i] 

    ##########
    edge_index = instance[0][0][0]
    x = instance[0][0][1]
    edge_attr = instance[0][0][2]
                                                                                                                    
    cuts = [sum([xcc.tolist() for xcc in xc], []) for xc in instance[1]]
                                                                                                                    
    nb_vars = len(cuts[0]) - 1 
    cut_instance_embeddings = []
    scores_cuts = []
    ##############################################################################################
    ################################## GNN call on instance ####################################
    for icut, cut in enumerate(cuts):
        x_cut = [[xx, cu, cut[-1]] for xx, cu in zip(x[:nb_vars], cut[:-1])] + [[xx, xx, xx] for xx in x[nb_vars:]]
        edge_index_formatted = torch.tensor(edge_index, dtype=torch.int64).T 
                                                                                                                    
        x_formatted = torch.tensor(x_cut, dtype=torch.float32)
        edge_attr_formatted = torch.tensor(edge_attr, dtype=torch.float32).reshape(len(edge_attr),1) 
                                                                                                                    
        instance_cut_embedding = torch.nn.functional.sigmoid(GNNModel(x_formatted, edge_index_formatted, edge_attr_formatted))
        scores_cuts.append(instance_cut_embedding[0].tolist()[0])

    winner_cut = np.argmax(scores_cuts)
    best_cut_index_nn = winner_cut
    #######################################################################################


    #pdb.set_trace()
    ##############################################################
    #best_cut_index_nn = np.argmax(scores_nn)        # without ranking
    ####################################################################
    best_indices_nn.append(best_cut_index_nn)
    best_lhs_nn = gom_cuts[0][best_cut_index_nn]
    best_rhs_nn = gom_cuts[1][best_cut_index_nn] 
    
    #x = instance_i 
    #x = raw_ip_instance_test[i]
    #instance_ = vector_to_ip(x[0], x[2], x[1], sense=sense)

    instance_ = graphToInstance(graph_i, sense=sense)
    instance_.add_cut(best_lhs_nn, best_rhs_nn)
    instance_.optimize()
    best_nn_tree_size = instance_.treesize
    best_nn_tree_sizes.append(best_nn_tree_size)

    #if i in [3,6]:
    #    pdb.set_trace()

if 0:
    t = np.linspace(0, len(best_nn_tree_sizes), len(best_nn_tree_sizes))
    import matplotlib.pyplot as plt
    plt.plot(t,best_nn_tree_sizes, label="NN")
    plt.plot(t, random_tree_sizes, label="random")
    plt.plot(t, parallelism_tree_sizes, label="parallelism")
    plt.plot(t, optimal_tree_sizes, label="optimal")
    plt.legend(loc="upper left")
    plt.show()

print("Nb of times parallelism is better than nn")
print(len([(i,j) for i,j in zip(best_nn_tree_sizes, parallelism_tree_sizes) if j < i]))

print("Nb of times nn is better than parallelism")  
print(len([(i,j) for i,j in zip(best_nn_tree_sizes, parallelism_tree_sizes) if i < j]))

print(np.mean(best_nn_tree_sizes))
print(np.mean(random_tree_sizes))
print(np.mean(parallelism_tree_sizes))
print(np.mean(optimal_tree_sizes))

pdb.set_trace()

