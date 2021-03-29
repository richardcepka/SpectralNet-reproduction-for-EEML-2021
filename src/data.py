from keras.models import model_from_json
import keras
from sklearn.model_selection import train_test_split
import torch
import scipy
import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import kneighbors_graph
import numpy as np
import scipy.sparse as sp
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import kneighbors_graph
import scipy.sparse


from pairs import create_pairs_from_unlabeled_data

def load_data(data, random_state=0):
    '''
    input:
    data - string
    
    return:
    X,y - numpy.ndarray, sparse
    '''
    if data == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        X,y = shuffle(X,y, random_state=random_state)
        X = X.reshape(-1,28*28).astype('float32') /255.     
        
    elif data == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        X,y = shuffle(X,y, random_state=random_state)
        X = X.reshape(-1,28*28).astype('float32') /255.

    else:
        raise ValueError('Name of dataset is invalid')

    return X,y

def get_embedding(X,json_file,h5_file):
    
    '''
    input:
    X - numpy.ndarray

    json_file: string
        - path for precomputed tf model
    h5_file: string
        - path for precomputed tf model
    
    return:
    X_e - numpy.ndarray
    '''
    #assert type(X) == np.ndarray, "X is not np.ndarray"
    json_file = open(json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_file)
    embending = keras.Model(inputs=loaded_model.input, outputs=loaded_model.layers[3].output)
    X_e = embending.predict(X)
    return X_e

class Dataset_Siemes(object):
    #numpy -> tensor
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index],
                self.label[index])

    def __len__(self):
        return self.size

def siamesedataloader(X, n_neighbors, test_size=0.2,batch_size=128,use_approx=False):
  '''
  input:
  X - numpy.ndarray
  
  n_neighbors: int
      -number of neighbors
  test_size: [0,1] int, default=0.2
      - size of test set
  batch_size: int, default=128
  
  return:
  train_dataloader,val_dataloader - torch dataloader
  '''
  #assert type(X) == np.ndarray, "X is not np.ndarray"
  
  pairs = create_pairs_from_unlabeled_data(X,k=n_neighbors,use_approx=use_approx)

  X_train, X_val, y_train, y_val = train_test_split(pairs[0], pairs[1], test_size=test_size)

  X1_train,X2_train,y_train = X_train[:,0,:],X_train[:,1,:],y_train
  X1_val,X2_val,y_val = X_val[:,0,:],X_val[:,1,:],y_val

  train_data = Dataset_Siemes(X1_train, X2_train, y_train)
  train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)

  val_data = Dataset_Siemes(X1_val, X2_val, y_val)
  val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=batch_size, shuffle=True)
  return train_dataloader, val_dataloader


def to_graph(X,sigma,e,n_neighbors,similarity_matrix,knn_aprox,eps=1e-7):
    '''
    Compute similarity matrix.

    return: similarity matrix
    '''
    if type(X) == torch.Tensor:
      X = X.detach().to("cpu").numpy()
      
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
