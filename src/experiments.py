import torch
import torchvision 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import keras

import numpy as np

from metrics import acc
from sklearn.metrics import normalized_mutual_info_score

from loss import SpectralNetLoss


from data import load_data
from data import get_embedding
from data import siamesedataloader
from data import to_graph


from train import train_AE, train_SiameseNet, train_SpectralNet


from sklearn.metrics import normalized_mutual_info_score

import time

from sklearn.model_selection import train_test_split

from spectral_methods import SpectralClustering
from sklearn.cluster import KMeans

'''
pip install keras==2.3
pip install tensorflow==1.15
pip install annoy
'''

def run_experiments(data=None, n_clusters=None, embedding=True, train_own_AE = False, siam=True, batch_size= 1024 , n_neighbors=2, aprox=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  X,y = load_data(data)

  if embedding:
    if train_own_AE:
      print("train AE")
      X_train, X_val = train_test_split(X, test_size=0.2)

      train_dataloader = torch.utils.data.DataLoader(torch.from_numpy(X_train), batch_size=128, shuffle=True)
      val_dataloader = torch.utils.data.DataLoader(torch.from_numpy(X_val), batch_size=128, shuffle=True)

      input_size = X.shape[1]
      verbose = True
      code_size = 10
      model = train_AE(input_size, code_size, train_dataloader,val_dataloader,verbose=verbose)

      X = model.encoder(torch.from_numpy(X).to(device))
      X = X.detach().to("cpu").numpy()

    else:
      #MNIST
      print("pretrain AE")
      json_file = "pretrain_AEweights\AE_mnist.json"
      h5_file = "pretrain_AEweights\AE_mnist_weights.h5"
      X = get_embedding(X, json_file, h5_file)
    
  
  #get siam model
  if siam:
    print("train siam")
    n_neighbors_siam = 2 
    use_approx = True
    train_dataloader, val_dataloader = siamesedataloader(X, n_neighbors_siam, use_approx=use_approx)

    input_size = X.shape[1] 
    verbose = True
    output_size_siam = 10
    model_siam = train_SiameseNet(input_size,output_size_siam,train_dataloader,val_dataloader, verbose=verbose)
  else:
    model_siam = None

  print("train SpectralNet")
  verbose=True
  model_spectralN = train_SpectralNet(n_clusters, X, batch_size, n_neighbors, aprox=aprox, model_siam=model_siam,verbose=verbose)
  
  
  print(X.shape)
  print("results: ")
  #SpectralNet
  Y = model_spectralN(torch.from_numpy(X).to(device))
  print("SpectralNet")
  kmeans = KMeans(n_clusters=n_clusters).fit(Y.detach().to("cpu").numpy())
  print(f"NMI = {normalized_mutual_info_score(y,kmeans.labels_)}")
  print(f"ACC = {acc(y, kmeans.labels_)}")

  #k-means
  print("k-means")
  if siam:
    Y = model_siam(torch.from_numpy(X).to(device))
    Y = Y.detach().to("cpu").numpy()
  
  else:
    Y = X
      
  kmeans = KMeans(n_clusters=n_clusters).fit(Y)
  print(f"NMI = {normalized_mutual_info_score(y,kmeans.labels_)}")
  print(f"ACC = {acc(y, kmeans.labels_)}")

  # SpectralClustering
  print("SpectralClustering")
  if siam:
    Y = model_siam(torch.from_numpy(X).to(device))
    Y = Y.detach().to("cpu").numpy()
  
  else:
    Y = X

  sc = SpectralClustering(n_clusters=n_clusters, n_neighbors=int(np.log(Y.shape[0])*3), sigma='mean', type_of_laplacian='clasic')
  y_labels = sc.fit(Y)
  print(f"NMI = {normalized_mutual_info_score(y,y_labels)}")
  print(f"ACC = {acc(y, y_labels)}")
  
  #return model_spectralN

if __name__ == "__main__":
     run_experiments(data="mnist", n_clusters=10, embedding=True, train_own_AE = False, siam=True, aprox=False)
  
