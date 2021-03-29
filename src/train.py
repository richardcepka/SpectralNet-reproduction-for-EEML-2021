import torch
from torch import nn
import torch.optim as optim

from loss import ContrastiveLoss
from loss import SpectralNetLoss

from nets import SiameseNet
from nets import AE
from nets import SpectralNet

from data import to_graph
from sklearn.model_selection import train_test_split

def train_AE(input_size,code_size,train_dataloader,val_dataloader,file = None,epochs = 200,verbose=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = AE(input_size=input_size,code_size=code_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = nn.MSELoss()

  for epoch in range(epochs):
      loss_t = 0
      model.train()
      for batch_features in train_dataloader:
          batch_features = batch_features.to(device)
          
          optimizer.zero_grad()
          
          outputs = model(batch_features)
          
          train_loss = criterion(outputs, batch_features)
          
          train_loss.backward()
          
          optimizer.step()
          
          loss_t += train_loss.item()

      loss_t = loss_t / len(train_dataloader)

      loss_v = 0
      model.eval()
      for batch_features in val_dataloader:
          batch_features = batch_features.to(device)
          
          outputs = model(batch_features)
          
          val_loss = criterion(outputs, batch_features)
          
          loss_v += val_loss.item()

      
      loss_v = loss_v / len(val_dataloader)
      scheduler.step(val_loss)
      act_lr = optimizer.param_groups[0]['lr']
      if verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, epochs,act_lr, loss_t, loss_v))
      if act_lr <= 1e-7:
        break
  if file!= None:
    torch.save(model, file)
  return model

def train_SiameseNet(input_size,output_size,train_dataloader,val_dataloader,file = None,epochs = 200,verbose=False):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = SiameseNet(input_size=input_size,output_size=output_size).to(device)
  optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
  criterion = ContrastiveLoss(1)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  for epoch in range(epochs):
      loss_t = 0
      model.train()
      for x1,x2,labels in train_dataloader:
          x1,x2,labels = x1.to(device),x2.to(device),labels.to(device)
          
          optimizer.zero_grad()
          
          z1, z2  = model(x1,x2)
          train_loss = criterion(z1, z2, labels)

          train_loss.backward()
          
          optimizer.step()
          
          loss_t += train_loss.item()

      loss_t = loss_t / len(train_dataloader)
      loss_v = 0
      model.eval()
      for x1,x2,labels in val_dataloader:
          x1,x2,labels = x1.to(device),x2.to(device),labels.to(device)
          
          z1, z2 = model(x1,x2)
          val_loss = criterion(z1, z2, labels)
          
          loss_v += val_loss.item()

      
      loss_v = loss_v / len(val_dataloader)
      scheduler.step(val_loss)
      act_lr = optimizer.param_groups[0]['lr']
      if verbose:
        print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, epochs,act_lr, loss_t, loss_v))
      if act_lr <= 1e-7:
        break
  if file!= None:
    torch.save(model, file)
  return model

def train_SpectralNet(output_size, X, batch_size, n_neighbors, aprox = False, model_siam=None, file = None, epochs = 100, verbose=False):

  #input -> X -numpy
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if model_siam == None:
    siam_metric = False
  else:
    siam_metric = True

  x_train, x_val = train_test_split(X, test_size=0.2)
  x_train = torch.from_numpy(x_train)
  x_val = torch.from_numpy(x_val)
  train_dataloader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=True)
    
  model = SpectralNet(input_size=x_train.shape[1], output_size=output_size).to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3) 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  criterion = SpectralNetLoss()

  for epoch in range(epochs):
    loss_t = 0
    for _ in train_dataloader:
      #ortostep
      model.eval()
      indices = torch.randperm(len(x_train))[:batch_size]
      x = x_train[indices].to(device)
      _ = model(x,ortho_step=True)

      #gradstep
      model.train()
      indices = torch.randperm(len(x_train))[:batch_size]
      x = x_train[indices].to(device)

      optimizer.zero_grad()

      Y = model(x)
      if siam_metric:
        x = model_siam(x)
      W = to_graph(x.detach().to("cpu").numpy(),"mean",None,n_neighbors,'k-hNNG',aprox).todense()

      W = torch.from_numpy(W).to(device)

      train_loss = criterion(Y,W)
      train_loss.backward()
              
      optimizer.step()
              
      loss_t += train_loss.item()

    loss_t = loss_t / len(train_dataloader)
    #valid
    model.eval()
    loss_v = 0
    for x in val_dataloader:
      x = x.to(device)
    
      Y = model(x)

      if siam_metric:
        x = model_siam(x)
      W = to_graph(x.detach().to("cpu").numpy(),"mean",None,n_neighbors,'k-hNNG',aprox).todense()

      W = torch.from_numpy(W).to(device)
            
      val_loss = criterion(Y,W)
            
      loss_v += val_loss.item()
      
    loss_v = loss_v / len(val_dataloader)
    scheduler.step(loss_v)
        
    act_lr = optimizer.param_groups[0]['lr']
    if verbose:
      print("epoch : {}/{}, learning_rate {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch + 1, epochs,act_lr, loss_t, loss_v))
    if act_lr <= 1e-7:
      break
  if file!= None:
    torch.save(model, file)
  return model
  

