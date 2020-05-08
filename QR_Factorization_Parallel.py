"""==========================================================================================================
The QR factorization of a matrix A is the decomposition of A into the product A = QR where Q
is an orthogonal matrix Q (or semi-orthogonal if A is not square)  ( t(Q)Q = I) and R upper triangular matrix.
QR decomposition is often used to solve the linear least squares problem and to determine the 
pseudo inverse of the matrix A. 
The linear systems AX =Y could be written QRX = Y or RX = t(Q)Y. The system resolution is faster and 
avoids the computation of the inverse of A. 
============================================================================================================"""


"""=========================================================================================================
Note that the QR decomposition is not unique (If A is a mxn matrix, you can take Q in the format mxm and R mxn 
or Q mxn and R nxn etc...)
A standard method to compute the QR decomposition uses Householder Transformations or Householder Reflections,
wich I will use here. It is known to be more numerically stable than the alternative Gramm-Schmidt method.
Please see the following website for more information about this method : 
https://en.wikipedia.org/wiki/Householder_transformation#QR_decomposition

A Householder Reflection is a linear transformation that enables a vector to be reflected through 
a plane or hyperplane. Essentially, we use this method because we want to create an upper triangular
matrix, R . The householder reflection is able to carry out this vector reflection such that all but
one of the coordinates disappears. The matrix Q
will be built up as a sequence of matrix multiplications that eliminate each coordinate in turn, 
up to the rank of the matrix A.

If we take a vector v, which in our case will be the kth column of A, we create a first reflection 
vector u1 with the following formula:

u1 = v  + a*e 
where e is the is the first column of the identity matrix I and a = -sign (vk)* ||v|| 
||.|| is the euclidean norm.

We convert u1 to a unit vector :

w1 = u1 / ||u1||

Then we form the Q matrix : 
Q1 = I - 2 w1*t(w1)

Finally, 
Q1*A creates a new matrix which has all zeroes
below the main diagonal in the first column.
The whole process is now repeated for the minor matrix 
, which will give a second Householder matrix , then a third ...

Once we have carried out all the iterations of this process we have R as an upper triangular matrix :
R = Q1 * Q2 * .....*A
Q is then fully defined as the multiplication of the transposes of each Qk. 
==========================================================================================================="""


"""=========================================================================================================
I propose here a parallel version of the QR factorization using MPI in python. 
The program consists in partitioning the matrix into blocks of a certain number of rows, each processor 
computing a partition. 
This algortithm will be tested on matrices called "tall and skinny" which suppose very large number of rows 
comparing to the number of columns which is often the case in linear regression problems.
A comparision will be made with the QR decomposition from Scipy and the sequential computation. 
==========================================================================================================="""


#coding: utf-8 
#command mpirun -n nb_procs python python_file.py
# Import mpi4py and call MPI
import mpi4py
import mpi4py.MPI as MPI
import numpy as np
from math import copysign, hypot, sqrt
import scipy
import scipy.linalg
import time
# Initialise the environnement MPI : automatically done when calling import mpi4py

#mpi4py.rc.initialize = False
#MPI.Init()



comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()


#We have to define a partition function to define local nrows
def partition(rank, size, N):
    n = N//size + ((N%size)>rank-1)
    s = (rank-1)*(N//size)
    if (N%size)>rank-1:
        s = s+rank-1
    else:
        s = s+N%size
    return s, s+n-1
    

def mult_matrix(M, N):
    return np.dot(M,N)

def trans_matrix(M):
    return M.transpose()

def norm(x):
    """Return the Euclidean norm of the vector x."""
    return np.linalg.norm(x)


def householder(A):
    """Perform QR decomposition of matrix A using Householder reflection."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    
    for k in range(num_cols - 1):
        x = R[k:, k]

        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -A[k, k])# take the value of the norm and the sign of the A[k,k] element
        u = x + e
        v = u / np.linalg.norm(u)

        Q_k = np.identity(num_rows)
        Q_k[k:, k:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_k, R)
        Q = np.dot(Q, Q_k.T)

    return Q, R
    
# We define our matrices
nrows = 1000
ncols = 100
A = np.empty([nrows, ncols],dtype=np.float64)
Q = np.empty([nrows,nrows], dtype = np.float64)
R = np.empty([nrows, ncols], dtype = np.float64)


"""=================================
Master : Proc 0 
Tasks : 
    *Generate Matrix A
    *Compute local nrows
    *Distribute the Matrix to slaves
    (Send local matrices)
==================================="""

if rank == 0:
    
    print("MPI is initialized", MPI.Is_initialized())
    print ("Running %d parallel MPI processes" % nb_procs)

    # Generate Matrix  : 
    A = np.random.rand(nrows, ncols).astype(np.float64)

    #Partition the Matrix and store the partitions at respective slaves
    #The proc 0 does not work(my choice), we distribute the calculations among the slaves
    for i in range(1, nb_procs):
        
        local_start_index, local_end_index = partition(i, nb_procs-1, nrows)

        #Compute and Send local nrows
        local_nrows = np.array([local_end_index+1-local_start_index],'i')
        comm.Send([local_nrows,MPI.INT], dest = i, tag = 0)
        
        #Compute and Send the local matrices to the slaves
        
        local_A = np.asarray(A[local_start_index:local_end_index+1])
        comm.Send([local_A,MPI.DOUBLE], dest = i, tag = 1)
        
        
#"""==================================================
#Slaves 
#Tasks :
#    * Create local nrows and receive it from master
#    * Create local A and receive it from master
#    * Create local Q and local R
#======================================================"""

if (rank>=1 and rank <=nb_procs-1) :
    print ("I am proc {} and I compute the partition number {}".format(rank, rank-1)) 
    
    #create and receive the local nrows
    local_nrows = np.array([1],'i')
    comm.Recv([local_nrows,MPI.INT], source = 0, tag=0)  
    #print("proc {} received local_nrows".format(rank))
    #print(local_nrows)
    
    #create and receive the local A
    nrows_local = local_nrows[0]
    local_A = np.empty([nrows_local, ncols], dtype=np.float64)
    comm.Recv([local_A,MPI.DOUBLE], source = 0, tag=1)
    #print("proc {} received local_matrix".format(rank))
    #print(local_matrix)
    
"""=============================================
The Slaves computes the local_Q and local_R
and send them to master
============================================"""

if (rank>=1 and rank<=nb_procs-1):
    #print("proc {} received vector".format(rank))
    
    # compute the local result at workers
    #print("proc {} starts computation".format(rank))
    #Time each proc compute the Matrix-Vector Multiplication
    mpi_start =time.time()
    local_Q, local_R = householder(local_A)
    #local_Q, local_R = scipy.linalg.qr(local_A) 
    mpi_end =time.time()
    print("proc {} local_Q.shape {}".format(rank, local_Q.shape))
    
    #Send y_local to master
    comm.Send([local_Q,MPI.DOUBLE], dest = 0, tag=2)
    comm.Send([local_R,MPI.DOUBLE], dest = 0, tag=3)

    #print("proc{} send local_y".format(rank))
    #Send time of computation 
    local_mpi_time = np.array([mpi_end-mpi_start], dtype=np.float64)
    comm.Send([local_mpi_time, MPI.DOUBLE], dest =0, tag =4)
    
"""===========================================
Master receive the Q_local and R_local 
gather all in R and Q
+ Check the results by computing the norm
+ Compare computational time 
==========================================="""
if rank ==0:
    total_mpi_time = 0
    x =0
    y= nrows
    for i in range(1,nb_procs):
        local_start_index, local_end_index = partition(i, nb_procs-1, nrows)
        local_nrows = np.array([local_end_index+1-local_start_index],'i')
        local_Q = np.empty([local_nrows[0],local_nrows[0]],dtype=np.float64)
        local_R = np.empty([local_nrows[0],ncols],dtype=np.float64)
        comm.Recv([local_Q, MPI.DOUBLE],source = i, tag = 2)
        comm.Recv([local_R, MPI.DOUBLE],source = i, tag = 3)
        
        local_mpi_time = np.empty([1], dtype=np.float64)
        comm.Recv([local_mpi_time, MPI.DOUBLE], source =i , tag =4)
        #Q[local_nrows[0]] = local_Q
        
        Q[local_start_index:local_end_index+1] = np.pad(local_Q, ((0, 0),(x,y-local_nrows[0]) ), 'constant')
        x += local_nrows[0]
        y -= local_nrows[0]
        R[local_start_index:local_end_index+1] = local_R
        #the total time of all procs to compute their partitions of the Matrix Vector Multiplication
        total_mpi_time += local_mpi_time
        #the maximum time used by a single proc
        average_mpi_time = total_mpi_time/(nb_procs-1)
        
    
    #with numpy
    start_scipy = time.time()
    Q_scipy, R_scipy = scipy.linalg.qr(A) 
    end_scipy = time.time()
    
    #Time sequential with 1 proc 
    start_seq = time.time()
    Q_seq, R_seq = householder(A)
    end_seq = time.time()
    #Check results : Frobenius norm of the difference
    print("Sequential norm [Q,R] :", [np.linalg.norm(Q_seq),np.linalg.norm(R_seq)])
    print("Sequential time :", end_seq-start_seq)
    print("MPI norm [Q,R]:", [np.linalg.norm(Q),np.linalg.norm(R)])
    #print("MPI total time of computation of all procs :", total_mpi_time[0])
    #print("MPI maximum time used by a single proc for computation :", max_mpi_time_proc[0])
    print("MPI average time used by a single proc for computation :", average_mpi_time[0])
    print("Scipy norm [Q,R] :", [np.linalg.norm(Q_scipy),np.linalg.norm(R_scipy)])
    print("Scipy time :", end_scipy-start_scipy)
    print (" MPI check A-QR = 0", np.linalg.norm(A-np.dot(Q,R)))
    print (" Scipy check A-QR = 0: ", np.linalg.norm(A-np.dot(Q_scipy,R_scipy)))
    print (" Sequential check A-QR = 0 :", np.linalg.norm(A-np.dot(Q_seq,R_seq)))
# Close the environnement MPI (automatically done)
#print (MPI.Is_finalized())
#mpi4py.rc.finalize = False
#MPI.Finalize()




    
    


        








