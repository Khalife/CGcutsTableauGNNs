# 1. Generate IP instances 

2 scripts

python generate\_set\_cover.py nb\_elements nb\_subsets nb\_samples

python generate\_uncapacited\_facility\_location.py nb\_customers nb\_facilities nb\_samples



# 2. Collect datasets

1 script

python collect\_data.py dataname N K nb\_samples file\_number\_samples cut\_score

dataname: set\_cover, facility

N: nb\_elements and nb\_subsets for set\_cover

K: nb\_customers and nb\_facilities for facility\_location

cut\_score: "gap" or "bc"


# 3. Train GNN

1 script


python train\_cg\_gnn.py N K nb\_samples dataname cut\_score inner\_dim\_1 inner\_dim\_2 nb\_iterations\_gnn 

Examples (used for Numerical experiments) 

python train\_cg\_gnn.py 10 10 1000 facility bc 5 5 5

python train\_cg\_gnn.py 30 50 1000 set\_cover gap 5 5 5


# 4. Test Phase

1 script

python test\_cg\_gnn.py N K nb\_samples dataname cut\_score inner\_dim\_1 inner\_dim\_2 nb\_iterations\_gnn  

Examples (used for numerical experiments)

python test\_cg\_gnn.py 10 10 1000 facility bc 5 5 5

python test\_cg\_gnn.py 30 50 1000 set\_cover gap 5 5 5

