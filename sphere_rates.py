import numpy as np
import graphlearning as gl
from scipy.special import gamma
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

def spherical_harmonics(x,y,z):

    V = np.vstack((np.ones_like(x),x,y,z,x**2-y**2,x*y,x*z,y*z,3*z**2-1)).T
    V = V / np.linalg.norm(V,axis=0)

    #Gram-Schmidt
    q,r = np.linalg.qr(V[:,1:4])
    V[:,1:4] = q
    q,r = np.linalg.qr(V[:,4:])
    V[:,4:] = q
    return  V


#Simulation on sphere
m = 2
alpha = np.pi**(m/2)/gamma(m/2+1)
alphaup = np.pi**((m+1)/2)/gamma((m+1)/2+1)
p = 1/(m+1)/alphaup #Density
val_exact = np.array([0,2,2,2,6,6,6,6,6])#,12,12,12,12,12,12,12,20,20,20,20,20,20]) #First 22 eigenvalues
num_vals = len(val_exact)
sigma = alpha/(m+2)


for e in range(12,18):
    n = 2**e #Number of points
    k = int(n**(4/(m+4))) #Number of nearest neighbors
    
    for T in range(100):

        #Random samples on sphere
        X = gl.utils.rand_ball(n,m+1)
        X = X / np.linalg.norm(X,axis=1)[:,np.newaxis]

        #knngraph 
        J,D = gl.weightmatrix.knnsearch(X,k)
        W = gl.weightmatrix.knn(None,k,knn_data=(J,D),kernel='uniform')
        L = (2*p**(2/m)/sigma)*gl.graph(W).laplacian()*((n*alpha/k)**(1+2/m))/n
        vals_knn,vecs_knn = eigsh(L,k=num_vals,which='SM')

        #Eps graph, reusing knnsearch from above
        eps = np.min(np.max(D,axis=1))
        mask = D.flatten() <= eps
        I = np.ones((n,k))*np.arange(n)[:,None]
        I = I.flatten()[mask]
        J = J.flatten()[mask]
        D = D.flatten()[mask] 
        W = coo_matrix((np.ones_like(D),(I,J)),shape=(n,n)).tocsr()
        L = (2/p/sigma)*gl.graph(W).laplacian()/(n*eps**(m+2))
        vals_eps,vecs_eps = eigsh(L,k=num_vals,which='SM')

        val_err_knn = np.absolute(val_exact - vals_knn)
        val_err_eps = np.absolute(val_exact - vals_eps)

        V = spherical_harmonics(X[:,0],X[:,1],X[:,2])
        vec_proj_knn = np.zeros(num_vals)
        vec_proj_eps = np.zeros(num_vals)
        vec_proj_knn[0]=1
        vec_proj_eps[0]=1

        for i in range(1,4):
            for j in range(1,4):
                vec_proj_knn[i] += np.sum(V[:,j]*vecs_knn[:,i])**2
                vec_proj_eps[i] += np.sum(V[:,j]*vecs_eps[:,i])**2

        for i in range(4,num_vals):
            for j in range(4,num_vals):
                vec_proj_knn[i] += np.sum(V[:,j]*vecs_knn[:,i])**2
                vec_proj_eps[i] += np.sum(V[:,j]*vecs_eps[:,i])**2

        print(T,end=',')
        print(n,end=',')
        print(k,end=',')
        print(eps,end=',')
        for i in range(num_vals):
            print(val_err_knn[i],end=',')
        for i in range(num_vals):
            print(val_err_eps[i],end=',')
        for i in range(num_vals):
            print(vec_proj_knn[i],end=',')
        for i in range(num_vals):
            print(vec_proj_eps[i],end=',')
        print('1',flush=True) 

