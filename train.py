import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.config import args
from src.utils import *
from src.dataset import *
# from src.transform import *

if __name__=="__train__":

    print("")

print("I am working")
images_df = create_data()
print(images_df.shape)

extra_df = create_extra_data()

images_df_final = images_df.copy()

labelencoder = LabelEncoder()
images_df["labels"] = labelencoder.fit_transform(images_df["labels"])

# Set seed for reproducibility
seed(args.seed, args.cuda)

# create directories
create_directories(args.save_dir)

model_ft = get_model(model_name = 'se_resnext101_32x4d')
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), 
                        lr=args.learning_rate, 
                        momentum=args.momentum)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

st_kfold = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

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


    model_ft = train_model(model_ft, criterion, 
                          optimizer_ft, 
                          exp_lr_scheduler, 
                          num_epochs=args.epochs,
                          dataloaders=dataloaders,
                          dataset_sizes=dataset_sizes
                          )

    break

# 
state = {'epoch': args.epochs,'state_dict': model_ft.state_dict(), 
            'optimizer': 'optimizer_ft.state_dict()', 
              'loss':'epoch_loss','valid_accuracy': 'best_acc'}

full_model_path = args.save_dir+'/model_state.tar'
torch.save(state, full_model_path)

print("All is done")