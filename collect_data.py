import pdb, sys
from GomoryCuts import *
import numpy as np
from env import *
from copy import copy, deepcopy
import pickle
import torch
from torch_geometric.data import Data

dataname = sys.argv[1]
N = int(sys.argv[2])
K = int(sys.argv[3])
nb_samples = int(sys.argv[4])

file_number_samples = nb_samples
instances = file_number_samples
#num_cons = N
#num_vars= K
prefix = "/local_workspace/khalsamm/"

if dataname == "set_cover":
    file_paths = [f"{prefix}/set_cover_{N}_{K}_num{instances}/set_cover_ip_data_{i}.npy" for i in range(instances)]

if dataname == "knapsack":
    file_paths = [f"{prefix}/knapsack_{N}_{K}_num{instances}/knapsack_ip_data_{i}.npy" for i in range(instances)]

if dataname == "facility":
    file_paths = [f"{prefix}/facility_{N}_{K}_num{instances}/uncapacitated_facility_location_ip_data_{i}.npy" for i in range(instances)]

def instanceToGraph1(IP_sample):
    A = IP_sample.A
    b = IP_sample.b
    c = IP_sample.c
    num_vars = IP_sample.num_vars
    num_cons = IP_sample.num_cons
    x = []
    edge_attr = []
    edge_index = []
    non_zeros = np.nonzero(A)
    for nzr, nzc in zip(non_zeros[0], non_zeros[1]):
        #adjacency_matrix.append([nzr+num_vars, num_vars])   
        edge_index.append([nzc, nzr + num_vars])
        #edge_attr.append(A[nzr, nzc])
        edge_attr.append(A[nzr, nzc])

    for i in range(num_vars):
        x.append(c[i])
        
    for i in range(num_cons):
        x.append(b[i])
    
    return [edge_index, x, edge_attr]


gnn_mode = 1
scores = []
instances = []
instances_graphs = []
indices_cuts_per_instance = []
cuts = []
data_list = []
ip_data_list = []
global_counter = -1
for sample in  range(nb_samples):
    print(sample)
    # not keeping the integer constraints
    if dataname == "knapsack":
        sense = "maximize"
    if dataname == "set_cover":
        sense = "minimize"
    if dataname == "facility":
        sense = "minimize"

    IP_sample = IP(filename=file_paths[sample], sense=sense)
    pdb.set_trace()
    A_base = IP_sample.A
    b_base = IP_sample.b
    x = ip_to_vector(IP_sample, mode="full")
    #pdb.set_trace()
    IP_sample.optimize()
    num_cons = A_base.shape[0]
    num_vars = A_base.shape[1]

    #pdb.set_trace()
    #print("objective without cut")
    #print(np.dot(IP_sample.c, IP_sample.x_IP))
    #pdb.set_trace()
    tree_size_before_cut = IP_sample.treesize
    ##### Generate Gomory's cuts #######
    #GMI_instance = GMI(IP_sample.A, c=IP_sample.c, b=IP_sample.b)  # obsolete
    #lhs_gom, rhs_gom = GMI_instance.candidate_GMI()                # obsolete
    lhs_gom, rhs_gom = IP_sample.get_candidate_rows()
    #pdb.set_trace()

    #pdb.set_trace()
    try:
        lhs_gom = lhs_gom[:, :IP_sample.num_vars]
        rhs_gom = rhs_gom.reshape(rhs_gom.shape[0],1)
    except:
        continue


    #pdb.set_trace()

    #if lhs_gom.shape[0] != 10:
    #    pdb.set_trace()
    nb_gomory_cuts = lhs_gom.shape[0]
    ip_data_list.append([x, IP_sample.num_vars, IP_sample.num_cons, lhs_gom, rhs_gom])
    if lhs_gom[0].shape[0] != IP_sample.num_vars:
        pdb.set_trace()
    ###################################
    #IP_tmp = deepcopy(IP_sample)
    indices_cuts_per_instance_sample = []
    for i in range(nb_gomory_cuts):
        #assert(A_base.shape[0] == 82) 
        global_counter += 1
        indices_cuts_per_instance_sample.append(global_counter)
        instances.append(x)
        #pdb.set_trace()
        IP_sample = vector_to_ip(x, num_cons, num_vars, sense=sense)
        #pdb.set_trace()
        #print(tree_size_before_cut)
        ##################################################################
        ##################################################################
        if gnn_mode:
            G = instanceToGraph1(IP_sample)
            instances_graphs.append(G)
        ##################################################################
        ##################################################################
        #pdb.set_trace()
        cut_lhs = lhs_gom[i,:] 
        cut_rhs = rhs_gom[i,:]
        #pdb.set_trace()
        #cuts.append(np.hstack([cut_rhs,cut_lhs]))
        IP_sample.add_cut(cut_lhs, cut_rhs) # add cut to instance
        IP_sample.optimize()
        #print("objective with cut")
        #pdb.set_trace()
        #print(np.dot(IP_sample.c, IP_sample.x_IP))
        tree_size_after_cut = IP_sample.treesize
        ###################################################################
        assert(len(np.where(~(np.dot(A_base, IP_sample.x_IP) <= b_base ))[0]) == 0) 
        #print([xx.getAttr("BranchPriority") for xx in IP_sample.model.getVars()])
        ####################################################################
        if tree_size_after_cut < tree_size_before_cut:
            print(f"Tree size after cut: {tree_size_after_cut}")
            print(f"Tree size before cut: {tree_size_before_cut}")

        #if tree_size_after_cut > tree_size_before_cut:
        #    pdb.set_trace()
        score = (tree_size_before_cut - tree_size_after_cut)/tree_size_before_cut
        scores.append(score)
        cuts.append([cut_lhs, cut_rhs])
        #IP_tmp = deepcopy(IP_sample)
        #IP_sample = IP(filename=file_paths[sample])
        #pdb.set_trace()
        
        ##############################################
        ####### Construct graph representation #######
        nb_vertices = num_vars + num_cons 
        edge_index = np.argwhere(IP_sample.A).tolist()
        
        x1 = IP_sample.c
        x2 = IP_sample.b 
        #x = np.hstack((x1, x2))
        #data_list.append(Data(x=x, edge_index=edge_index.t().contiguous()))

        #### take edges in account in message function
        

        #############################################
        ##############################################
    indices_cuts_per_instance.append(indices_cuts_per_instance_sample)

#pdb.set_trace()
#with open("knapsack_small_sample.pkl", "wb") as f:
    #pickle.dump([instances, cuts,scores], f)

#if nb_samples == 10:
#    torch.save([instances, cuts, scores, nb_cuts_per_instance], "knapsack_small_sample.pkl")
#    torch.save(ip_data_list, "knapsack_ip_data_list_small_sample.pkl")
#else:
torch.save([instances, cuts, scores, indices_cuts_per_instance], f"{prefix}/set_cover_{nb_samples}_samples.pkl")
torch.save(ip_data_list, f"{prefix}/set_cover_ip_data_list_{nb_samples}_samples.pkl".format(nb_samples))


if gnn_mode:
    torch.save([instances_graphs, cuts, scores, indices_cuts_per_instance], f"{prefix}/set_cover_{nb_samples}_graph_samples.pkl")
    torch.save(ip_data_list, f"{prefix}/set_cover_ip_data_list_{nb_samples}_graph_samples.pkl")
