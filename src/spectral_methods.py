import numpy as np 
from scipy import sparse
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import time
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph
import pynndescent
from pynndescent import PyNNDescentTransformer

class SpectralClustering:

  def __init__(self, n_clusters=2, n_neighbors=10, sigma='max', e=5, affinity='k-hNNG' , type_of_laplacian='rw',knn_aprox=True, eigen_aprox=True ,normalize=False, print_time=False):
    '''
    n_clusters : int, default=2
      The dimension of the projection subspace. Number of clusters.

    
    affinity : {'e-NG','full','k-hNNG','k-NNG','k-mNNG','precomputed'}, default='k-hNNG'
      How to construct the affinity matrix.
        - 'e-NG' :
        - 'k-hNNG' :
        - 'k-NNG' : 
        - 'k-mNNG' :
        ! - 'full' : construct the fully connected similarity matrix using a radial basis function
           (RBF) kernel.
        - ‘precomputed’ : interpret X as a precomputed affinity matrix.
    
    
    n_neighbors : int, default=10
      Number of neighbors to use when constructing the affinity matrix using
      the nearest neighbors method. 
      Ignored for affinity='full','e-NG',precomputed’

    
    sigma : int or str, default='max'
      Coefficient for (RBF).
      'max' - local parameter: maximal distance of n_neighbors
      ! 'mean' - local parameter: mean distance of n_neighbors
      Ignored for affinity='e-NG',precomputed’.

    e : int, default=5
      Radius of neighborhoods.
      Ignored for affinity='k-hNNG','k-NNG','k-mNNG','full', 'precomputed’.
      
    type_of_laplacian : {'clasic','sym','rw'}, default='rw'
      - 'clasic' : L = D-A
      - 'sym' : L_sym = D**(-1/2)@L@D**(-1/2) = I - D**(-1/2)@A@ D**(-1/2))
      - 'rw' : L_rw = D**(-1)@L = I - D**(-1)@A

    knn_aprox: bool, default=True
      
    self.normalize: bool, default=False
    
    print_time : {True,False}, default=False
      Prints the computation time of the to_graph, laplacian matrix, eigenvectors.
      
      ...
    '''
    self.n_clusters = n_clusters
    self.n_neighbors = n_neighbors
    self.sigma = sigma
    self.e = e
    self.type_of_laplacian = type_of_laplacian
    self.knn_aprox = knn_aprox
    self.affinity= affinity
    self.normalize = normalize
    self.print_time = print_time
    self.eigen_aprox = eigen_aprox

  def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    

  def compute_k_smallest_eig(self,L,n_clusters,eigen_aprox):
    '''
    Compute k smallest eigenvectors and eigenvalues.

    '''
    if eigen_aprox:
      eigenvalues, eigenvectors = sparse.linalg.eigsh(L,k=n_clusters, which='SM')
    else:
      eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
      values_idx = np.argsort(eigenvalues)
      eigenvectors = eigenvectors[:, values_idx[:n_clusters]]
      eigenvalues = eigenvalues[values_idx[:n_clusters]]
      
    
    "noramlize"
    if self.normalize:
      l2norm = norm(eigenvectors, axis=1, ord=2)
      eigenvectors = eigenvec.astype(np.float) / l2norm[:,None]

    return eigenvalues[1:], eigenvectors[:,1:]
  
  def fit(self,X):
    '''

    return: labels
    '''
    if self.print_time: start_time = time.time()
    self.A = self.to_graph(X,self.sigma,self.e,self.n_neighbors,self.affinity,self.knn_aprox)
    if self.print_time: print("to_graph %s seconds" % (time.time() - start_time))
    
    
    if self.print_time: start_time = time.time()
    L = self.laplacian(self.A,self.type_of_laplacian)
    if self.print_time: print("laplacian %s seconds" % (time.time() - start_time))
        
    
    if self.print_time: start_time = time.time()
    self.eigenval, self.eigenvec = self.compute_k_smallest_eig(L,self.n_clusters,self.eigen_aprox)
    if self.print_time: print("k smallest eigenvectors %s seconds" % (time.time() - start_time))
        
    
    if self.n_clusters == 2:
        self.labels = np.sign(self.eigenvec)>0
    else:
        kmeans = KMeans(n_clusters=self.n_clusters).fit(self.eigenvec)
        self.labels =  kmeans.labels_
    return self.labels


  def laplacian(self,A,type_of_laplacian):
    '''
    Compute Laplacian matrix matrix.

    return: graph of laplacian
    '''
    
    d = np.asarray(np.sum(A,axis=1),dtype=np.float32).reshape(-1)
    if type_of_laplacian =='clasic':
      return sparse.diags(d,dtype=np.float32) - A

    elif type_of_laplacian =='sym':
      _D_ = sparse.diags(d**(-1/2),dtype=np.float32)
      return _D_@(sparse.diags(d,dtype=np.float32) - A)@_D_
    
    elif type_of_laplacian=='rw':
      D_ = sparse.diags(d**(-1),dtype=np.float32)
      return D_@(sparse.diags(d,dtype=np.float32) - A)

  def to_graph(self,X,sigma,e,n_neighbors,similarity_matrix,knn_aprox,eps=1e-7):
    '''
    Compute similarity matrix.

    return: similarity matrix
    '''
    if similarity_matrix == 'e-NG':
      A = radius_neighbors_graph(X, e, mode='connectivity',include_self=False, n_jobs=-1)
      return A
    
    elif similarity_matrix == 'full':
        pass
    
    elif similarity_matrix == 'precomputed':
      return A

    else:
        
        if knn_aprox:
            A = PyNNDescentTransformer(n_neighbors=n_neighbors,metric="euclidean",n_jobs=-1).fit_transform(X)
        else:
            A = kneighbors_graph(X, n_neighbors, mode='distance',include_self=False, n_jobs=-1)
            
        if sigma == 'max':
            sigma_2 = 2*np.power(A.max(axis=1).toarray(),2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        elif sigma == 'mean':
            sigma_2 = 2*np.power(A.sum(axis=1) / A.getnnz(axis=1).reshape(-1,1),2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        else:
            sigma_2 = 2*np.power(sigma,2) + eps
            A = ( ( A.power(2,dtype=np.float32) ).multiply(1/sigma_2) ).tocsr()
            np.exp(-A.data,out=A.data)
        
        if knn_aprox:
            A = A - sparse.identity(A.shape[0])

        if similarity_matrix == 'k-hNNG':
            return (A + A.T)/2
            
        if similarity_matrix == 'k-NNG':
            return A.maximum(A.T)

        if similarity_matrix == 'k-mNNG':
            return A.minimum(A.T)
