
from torchvision import transforms as T

from randaugment import RandAugment

from .config import args

train_trans = T.Compose([
        T.RandomResizedCrop(args.size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(args.mean, args.std)
    ])

val_trans = T.Compose([
        T.Resize(500),
        T.CenterCrop(args.size),
        T.ToTensor(),
        T.Normalize(args.mean, args.std)
    ])

test_trans = T.Compose([
        T.Resize(500),
        T.CenterCrop(args.size),
        T.ToTensor(),
        T.Normalize(args.mean, args.std)
    ])


weak_trans =  T.Compose([
        T.Resize(448),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(size=448, 
                      padding=int(448*0.125), 
                      padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(args.mean, args.std)
    ])


strong_trans =  T.Compose([
        T.Resize(448),
        # T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(size=448, 
                      padding=int(448*0.125), 
                      padding_mode='reflect'),
      
        # T.RandomErasing(p=1, 
        #                 ratio=(1, 1), 
        #                 scale=(0.01, 0.01), 
        #                 value=127),
        RandAugment(),
        
        T.ToTensor(),
        T.Normalize(args.mean, args.std)    
    ])