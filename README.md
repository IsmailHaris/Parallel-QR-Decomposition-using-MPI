# Parallel QR Decomposition using MPI

Parallel computation of the QR Decomposition of a random rectangular matrix using the Householder reflections.

Using MPI in python. 

Comparaision with the sequential computation and the QR factorization from Scipy. 

The QR factorization of a matrix A is the decomposition of A into the product A = QR where Q
is an orthogonal matrix Q (or semi-orthogonal if A is not square)  ( t(Q)Q = I) and R upper triangular matrix.
QR decomposition is often used to solve the linear least squares problem and to determine the 
pseudo inverse of the matrix A. 

The linear systems AX =Y could be written QRX = Y or RX = t(Q)Y. The system resolution is faster and 
avoids the computation of the inverse of A.

Note that the QR decomposition is not unique (If A is a mxn matrix, you can take Q in the format mxm and R mxn 
or Q mxn and R nxn etc...)
A standard method to compute the QR decomposition uses Householder Transformations or Householder Reflections,
wich I will use here. It is known to be more numerically stable than the alternative Gramm-Schmidt method.
