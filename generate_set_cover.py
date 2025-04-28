import numpy as np
import os, pdb, sys
from IP import *

def generateSubsets(n, m, sparsity_level=[0.05, 0.1]):
    # n size of the universe 
    # m number of subsets
    # sparsity level: max probability of drawing an element for a subset 
    condition_subset_complete = False

    Rows = []
    while 1:
        proba_i = sparsity_level[0] + (sparsity_level[1]-sparsity_level[0])*np.random.rand() 
        row_i = np.random.binomial(size=n, n=1, p=proba_i).tolist()
        Rows.append(row_i)
        if len(Rows) == m:
            #pdb.set_trace()
            if 0 not in [sum([row[i] >= 1 for row in Rows]) for i in range(n)]: 
                print([sum([row[i] >= 1 for row in Rows]) for i in range(n)])
                return np.array(Rows) 
            else:
                Rows = []
        

def indicatorRowsToMatrixConstraint(Rows):
    # number of decision variables: number rows of Rows
    nb_subsets = Rows.shape[0]
    nb_elements = Rows.shape[1]
    nb_vars = Rows.shape[0]
    nb_cons = Rows.shape[1] + 2*nb_vars

    #pdb.set_trace()
    A = np.zeros((nb_cons, nb_vars))
    b = np.zeros(nb_cons)
    c = np.ones(nb_vars)
    #c = np.random.rand(nb_vars)
    for i in range(Rows.shape[1]):
        indices_i = np.where(Rows[:,i] == 1) 
        for index in indices_i[0]:
            A[i, index] = -1
            b[i] = -1
   
    ## variables between 0 and 1
    for i in range(nb_vars):
        A[nb_elements+i, i] = -1
        A[nb_elements+nb_vars+i, i] = 1
        b[nb_elements+i] = 0
        b[nb_elements+nb_vars+i] = 1
    
    #pdb.set_trace()
    return A, b, c
    #return -A.transpose(), c, -b 


if __name__ == "__main__":
    #Rows = generateSubsets(3, 6)
    #A,b,c = indicatorRowsToMatrixConstraint(Rows)


    #A_tmp = -A.transpose()                                                                                                
    #b_tmp = c                                                                                                             
    #c_tmp = -b


    #IPX = IP(A,c,b)
    #IPX.optimize()
    #print(np.array(get_simplex_tableau(A,c,b)).astype(float))
    #pdb.set_trace()

    nb_elements = int(sys.argv[1])
    nb_subsets = int(sys.argv[2])
    nb_samples = int(sys.argv[3])
    pathset = f"/local_workspace/khalsamm/set_cover_{nb_elements}_{nb_subsets}_num{nb_samples}/"
    if not(os.path.exists(pathset)):
        os.mkdir(pathset)
        for i in range(nb_samples):
            print(i)
            Rows = generateSubsets(nb_elements, nb_subsets, sparsity_level=[0.2,0.4])
            A,b,c = indicatorRowsToMatrixConstraint(Rows)
            #pdb.set_trace()
            np.save(f"/local_workspace/khalsamm/set_cover_{nb_elements}_{nb_subsets}_num{nb_samples}/set_cover_ip_data_{i}.npy", {'A':A, 'c': c, 'b': b})

