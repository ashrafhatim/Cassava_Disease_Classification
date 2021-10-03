import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import  models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import os
import random
import time
import copy
import tqdm


from .config import args
from transform import *
from model import *
from .dataset import CassavaDataset, extra_CassavaDataset


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    """
    function to train the model
    --
    INPUTS:
    model: (model object) 
    criterion : criterion for the loss
    optimizer : (optimizer object)
    scheduler : (scheduler object) to schedule the learning rate
    num_epochs: (int) number of epoch to train 
    --
    OUTPUTS: trained model
    """ 
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                 # 
                state = {'epoch': epoch,'state_dict': model.state_dict(), 
                            'optimizer': 'optimizer_ft.state_dict()', 
                             'loss':'epoch_loss','valid_accuracy': 'best_acc'}
                create_directories(args.save_dir)
                full_model_path = args.save_dir+'/model_state_fixmatch.tar'
                #full_model_path = saved_dir+'model_state.tar'
                torch.save(state, full_model_path)
                #
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train_all_model(model, 
                    criterion, 
                    optimizer, 
                    scheduler, 
                    dataloaders,
                    dataset_sizes,
                    num_epochs=20):
    """
    function to train the model
    --
    INPUTS:
    model: (model object) 
    criterion : criterion for the loss
    optimizer : (optimizer object)
    scheduler : (scheduler object) to schedule the learning rate
    num_epochs: (int) number of epoch to train 
    --
    OUTPUTS: trained model
    """ 
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in dataloaders:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 
                state = {'epoch': epoch,'state_dict': model.state_dict(), 
                            'optimizer': 'optimizer_ft.state_dict()', 
                             'loss':'epoch_loss','valid_accuracy': 'best_acc'}
                create_directories(args.save_dir)
                full_model_path = args.save_dir+'/model_state_pseudoLabling.tar'
                #full_model_path = saved_dir+'model_state.tar'
                torch.save(state, full_model_path)
                #

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def alpha_weight(step):
    if step < args.T1:
        return 0.0
    elif step > args.T2:
        return args.af
    else:
         return ((step-args.T1) / (args.T2-args.T1))*args.af
        

def semisup_train(model, unlabeled_loader, images_df,criterion, optimizer, st_kfold , scheduler):
  # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
  EPOCHS = 2
  
  # Instead of using current epoch we use a "step" variable to calculate alpha_weight
  # This helps the model converge faster
  step = 100 
  main_epoch = 0
  
  model.train()
  for epoch in tqdm.notebook.tqdm(range(EPOCHS)):
    print('epoch=', epoch, 'of unlabled data!')
    for batch_idx, x_unlabeled_tuple in enumerate(unlabeled_loader):

      x_unlabeled = x_unlabeled_tuple[0]
        
        
      # Forward Pass to get the pseudo labels
      x_unlabeled = x_unlabeled.cuda()
      model.eval()
      output_unlabeled = model(x_unlabeled)
      _, pseudo_labeled = torch.max(output_unlabeled, 1)
      model.train()          
      
      # Now calculate the unlabeled loss using the pseudo label
      output = model(x_unlabeled)
      unlabeled_loss = alpha_weight(step) * F.nll_loss(output, pseudo_labeled)   
      
      # Backpropogate
      optimizer.zero_grad()
      unlabeled_loss.backward()
      optimizer.step()
      
      
      # For every 50 batches train one epoch on labeled data 
      if batch_idx % 629 == 0:
        print('main epoch=',  main_epoch)

     
        # # Normal training procedure
        # for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        #     X_batch = X_batch.cuda()
        #     y_batch = y_batch.cuda()
        #     output = model(X_batch)
        #     labeled_loss = F.nll_loss(output, y_batch)

        #     optimizer.zero_grad()
        #     labeled_loss.backward()
        #     optimizer.step()
        

        fold = 0
        for train_index, val_index in st_kfold.split(images_df['images'], images_df['labels']):
          print('inside')

          train, val = images_df.iloc[train_index], images_df.iloc[val_index]

          train_dataset = CassavaDataset(df_data=train, transform=train_trans)
          valid_dataset = CassavaDataset(df_data=val,transform=val_trans)

          train_loader = DataLoader(dataset = train_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=2,
                                    )
          valid_loader = DataLoader(dataset = valid_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False, 
                                    num_workers=2,
                                    )

          dataloaders = {'train': train_loader, 'val': valid_loader}
          
          dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)}

          if fold == 1:
            model = train_model(model, criterion, 
                            optimizer, 
                            scheduler, 
                            num_epochs=1,)
          fold += 1
    
        # Now we increment step by 1
        step += 1
        main_epoch += 1

def Stratified_training(images_df,criterion, optimizer_ft, exp_lr_scheduler ):

  st_kfold = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

  fold = 0
  for train_index, val_index in st_kfold.split(images_df['images'], images_df['labels']):
      train, val = images_df.iloc[train_index], images_df.iloc[val_index]

      train_dataset = CassavaDataset(df_data=train, transform=train_trans)
      valid_dataset = CassavaDataset(df_data=val,transform=val_trans)

      train_loader = DataLoader(dataset = train_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=2,
                                )
      valid_loader = DataLoader(dataset = valid_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=2,
                                )

      dataloaders = {'train': train_loader, 'val': valid_loader}
      
      dataset_sizes = {'train': len(train_dataset), 'val': len(valid_dataset)}
      print(dataset_sizes)
      if fold == 1:

        saved_dir = args.save_dir+str(fold)+'/'

        print(f'Starting CV for Fold {fold}')

        model_ft = train_model(model_ft, criterion, 
                            optimizer_ft, 
                            exp_lr_scheduler, 
                            num_epochs=1,)

      fold += 1

def fix_match(model, criterion, optimizer , scheduler, st_kfold, images_df, extra_df, num_of_epochs = 2, threshold = 0.9):

  # gc.collect()
  # torch.cuda.empty_cache()
  # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
  EPOCHS = num_of_epochs
  
  # Instead of using current epoch we use a "step" variable to calculate alpha_weight
  # This helps the model converge faster
  
  model.train()
  for epoch in tqdm.notebook.tqdm(range(EPOCHS)):
    print('epoch=', epoch, '!!!')

    for train_index, val_index in st_kfold.split(images_df['images'], images_df['labels']):

      train, val = images_df.iloc[train_index], images_df.iloc[val_index]

      valid_dataset = CassavaDataset(df_data=val,transform=val_trans)
      valid_loader = DataLoader(dataset = valid_dataset, 
                                batch_size=4, 
                                shuffle=False, 
                                num_workers=2,
                                )

      l_train_dataset = CassavaDataset(df_data=train, transform=train_trans)
      l_train_loader = DataLoader(dataset = l_train_dataset, 
                                batch_size=4, 
                                shuffle=True, 
                                num_workers=2,
                                )
      

      u_train_dataset_weak = extra_CassavaDataset(df_data=extra_df, transform=weak_trans)
      u_train_loader_weak = DataLoader(dataset = u_train_dataset_weak, 
                                batch_size=11, 
                                shuffle=True, 
                                num_workers=2,
                                )

      u_train_dataset_strong = extra_CassavaDataset(df_data=extra_df, transform=strong_trans)
      u_train_loader_strong = DataLoader(dataset = u_train_dataset_strong, 
                                batch_size=11, 
                                shuffle=True, 
                                num_workers=2,
                                )
      
      for t1, t2, t3 in zip(l_train_loader, u_train_loader_weak, u_train_loader_strong):
        labeled_imgs, labels = t1[:][0], t1[:][1]
        weak_imgs, strong_imgs = t2[:][0], t3[:][0]

        model.eval()
        pseudo_labeled = model(weak_imgs.to(device=args.device))

        print(pseudo_labeled.shape)

        break
      break


def seed(seed, cuda):
    """
    function to fix the seed
    --
    INPUTS:
    seed: (int) the seed that we wanted to fix
    cuda: (bool) if we are using gpu , then also fix the seed related to cuda 
    --
    OUTPUTS: no output
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_directories(dir_path):
    """
    function to create directory for checkpoints
    --
    INPUTS:
    dir_path: (str) path of the directory
    --
    OUTPUTS: no output
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def filter_extra_imgs(model_ft, extra_df, threshold = 0.9):
    """
    function to filter the extra images given a threshold
    --
    INPUTS:
    threshold: (int) 
    --
    OUTPUTS: data frame contains the filtered images 
    """
    extra_paths = []
    peseudo_labels = []

    for img_path in extra_df['images']:
        img = Image.open(img_path)
        img = weak_trans(img)
        model_ft.eval()
        logits = model_ft(img.unsqueeze_(0).to(device=args.device))
        model_ft.train()
        if logits.max(-1).values.item() > threshold:
            extra_paths.append(img_path)
        peseudo_labels.append(torch.argmax(logits).item())

    df = pd.DataFrame(columns=['images', 'labels'])
    df['images'] = extra_paths
    df['labels'] = peseudo_labels

    return df

def load_checkpoint(model, optimizer, filename=None):
    """
    function to load the saved checkpoints (to continue the experiment)
    --
    INPUTS:
    model: (model object) 
    optimizer: (optimizer object)
    filename : (str)
    --
    OUTPUTS: model , optimizer , start epoch
    """ 
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states. 
    start_epoch = 0 
    if os.path.isfile(filename): 
        print("=> loading checkpoint '{}'".format(filename)) 
        checkpoint = torch.load(filename) 
        start_epoch = checkpoint['epoch'] 
        model.load_state_dict(checkpoint['state_dict']) 
        #optimizer.load_state_dict(checkpoint['optimizer']) 
        print("=> loaded checkpoint '{}' (epoch {})" .format(filename,
                                                            checkpoint['epoch'])) 
    else: print("=> no checkpoint found at '{}'".format(filename)) 
    return model, optimizer, start_epoch

def get_model(model_name = 'se_resnext101_32x4d'):
    """
    function to get model given the model name and adopted to cassava classification task
    --
    INPUTS:
    model_name: (str) 
    --
    OUTPUTS: model
    """ 
    if model_name == 'se_resnext101_32x4d':
        base_model = se_resnext101_32x4d(pretrained=True)
        model_ft = SEResnext101(base_model, 5)

    elif model_name == 'se_resnext50_32x4d':
        model_ft = se_resnext50_32x4d(pretrained=False)
        model_ft.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = torch.nn.Linear(num_ftrs, 5)

    elif model_name == 'resnet50':
        base_model = models.resnet50(pretrained=True)
        model_ft = ResNet50(base_model, 5)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 5)

    model_ft = model_ft.to(args.device)
    return model_ft