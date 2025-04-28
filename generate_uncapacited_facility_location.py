import numpy as np
import os, pdb, sys



def generateMatrices(nb_customers, nb_facilities, max_demand_per_customer, min_price, max_price, max_cost,ratio=1.5):
    #D = np.random.randint(0, max_demand, nb_customers)
    D = max_demand_per_customer*np.random.rand(nb_customers)  
    F = min_price + max_price*np.random.rand(nb_facilities)
    #U = ratio*(max_demand_per_customer)*(nb_customers/nb_facilities)*np.random.rand(nb_facilities) 
    M = nb_customers
    C = max_price*np.random.rand(nb_facilities, nb_customers) 

    nb_vars = nb_customers*nb_facilities + nb_facilities
    nb_cons = nb_facilities+2*nb_customers + 2*nb_vars 
    
    M = nb_customers

    c = np.zeros(nb_vars)
    for i in range(nb_facilities):
        for j in range(nb_customers):
            c[i*nb_customers + j] = C[i, j]*D[j] 

    for i in range(nb_facilities):
        c[nb_customers*nb_facilities + i] = F[i]

    
    A = np.zeros((nb_cons, nb_vars))
    b = np.zeros(nb_cons)
    #################################################
    ############# First equality constraints #########
    for i in range(nb_customers):
        for j in range(nb_facilities):
            A[i, j*nb_customers+i] = 1 
        b[i] = 1

    for i in range(nb_customers):
        for j in range(nb_facilities):
            A[nb_customers+i, j*nb_customers+i] = -1 
        b[nb_customers + i] = -1
    
    #################################################
    ############## Second group: inequalities ########


    for i in range(nb_facilities):
        for j in range(nb_customers):
            A[2*nb_customers+i, j + i*nb_customers] = 1
        A[2*nb_customers+i, nb_customers*nb_facilities+i] = -M    
        b[2*nb_customers+i] = 0



    ##################################################
    ############## Constraints between 0 and 1 #####
    for i in range(nb_vars):
        A[2*nb_customers + nb_facilities + i, i] = -1
        b[2*nb_customers + nb_facilities + i] = 0

    for i in range(nb_vars):
        A[2*nb_customers + nb_facilities + nb_vars + i, i] = 1
        b[2*nb_customers + nb_facilities + nb_vars + i] = 1
    ##################################################

    #pdb.set_trace()



    ######################################
    #########################################
    return A, b, c
    #return -A.transpose(), c, -b

if __name__ == "__main__":
    nb_facilities = 20
    nb_customers = 20
    max_demand_per_customer = 1.5*nb_facilities/nb_customers
    min_price = 5
    max_price = 100
    max_cost = 50

    A, b, c  = generateMatrices(nb_customers, nb_facilities, max_demand_per_customer, min_price, max_price, max_cost)
    print("c")
    print(b)
    print("b")
    print(-c)
    print("A")
    print(-A.transpose())
    #pdb.set_trace()

    nb_customers = int(sys.argv[1])
    nb_facilities = int(sys.argv[2])
    nb_samples = int(sys.argv[3])

    pathset = f"/local_workspace/khalsamm/uncapacitated_facility_location_{nb_customers}_{nb_facilities}_num{nb_samples}/"
    #if not(os.path.exists(pathset)):
    if 1:
        #os.mkdir(pathset)
        for i in range(nb_samples):
            print(i)
            A, b, c = generateMatrices(nb_customers, nb_facilities, max_demand_per_customer, min_price, max_price, max_cost)
            #pdb.set_trace()
            np.save(f"/local_workspace/khalsamm/uncapacitated_facility_location_{nb_customers}_{nb_facilities}_num{nb_samples}/uncapacitated_facility_location_ip_data_{i}.npy", {'A': A, 'c': c, 'b': b})

