import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os

from glob import glob

from .config import args

class CassavaDataset(Dataset):
    def __init__(self, df_data, transform=None):
        super().__init__()
        self.df = df_data.values
        
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path, label = self.df[index]
        
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

class extra_CassavaDataset(Dataset):
    def __init__(self, df_data, train_transform=None, strong_trans=None):
        super().__init__()
        self.df = df_data.values
        
        self.train_transform = train_transform
        self.strong_trans = strong_trans

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path, label = self.df[index]
        
        image = Image.open(img_path)

        if img_path.split('/')[1] == 'train':
          if self.train_transform is not None:
              image = self.train_transform(image)
        else:
          if self.strong_trans is not None:
              image = self.strong_trans(image)

        return image, label

class CassavaTestDataset(Dataset):
    def __init__(self, df_data, transform=None, tta=False, tta_idx=0, data_path='./test/0/'):
        super().__init__()
        self.df = df_data.values
        self.transform = transform
        self.tta=tta
        self.tta_idx = tta_idx
        self.data_path = data_path

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_name = self.df[index]
        img_path = os.path.join(self.data_path, image_name)
        image = Image.open(img_path)
        if self.tta:
           image = crop_image(image, crop_idx=self.tta_idx)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


def get_labels(file_path): 
    """
    function to get labels 
    --
    INPUTS:
    file_path: (str) the path of the images
    --
    OUTPUTS: label from the path example (healthy,cgm,cmd,cbsd,cbb)
    """
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)

def create_data():
  imagePaths = glob("./data/train/*/*", recursive=True)
  print(imagePaths)
  images_df = pd.DataFrame(columns=['images', 'labels'])
  images_df["images"] = imagePaths

  labels = []
  for img in imagePaths:
      labels.append(get_labels(img))   

  images_df["labels"] = labels

  return images_df

def create_extra_data():
  extra_imagePaths = glob("./data/extraimages/*.*", recursive=True)
  extra_df = pd.DataFrame(columns=['images', 'labels'])
  extra_df["images"] = extra_imagePaths

  labels = []
  for img in extra_imagePaths:
      labels.append(-3)   

  extra_df["labels"] = labels

  return extra_df

def crop_image(im, crop_idx):
    """
    function to perform image croping and flipping (image augmentation)
    --
    INPUTS:
    im: (image object) 
    crop_idx : the index of cropping 
    --
    OUTPUTS: augmented image
    """ 
    w, h = im.size
    if crop_idx == 0:
        im = im.crop((0, 0, int(w*0.9), int(h*0.9))) # top left
    elif crop_idx == 1:
        im = im.crop((int(w*0.1), 0, w, int(h*0.9))) # top right
    elif crop_idx == 2:
        im = im.crop((int(w*0.05), int(h*0.05), w-int(w*0.05), h-int(h*0.05))) # center
    elif crop_idx == 3:
        im = im.crop((0, int(h*0.1), w-int(w*0.1), h)) # bottom left
    elif crop_idx == 4:
        im = im.crop((int(w*0.1), int(h*0.1), w, h)) # bottom right
    elif crop_idx == 5:
        im = im.crop((0, 0, int(w*0.9), int(h*0.9))) 
        im = im.transpose(Image.FLIP_LEFT_RIGHT) # top left and HFlip
    elif crop_idx == 6:
        im = im.crop((int(w*0.1), 0, w, int(h*0.9)))
        im = im.transpose(Image.FLIP_LEFT_RIGHT) # top right and HFlip
    elif crop_idx == 7:
        im = im.crop((int(w*0.05), int(h*0.05), w-int(w*0.05), h-int(h*0.05)))
        im = im.transpose(Image.FLIP_LEFT_RIGHT) # center and HFlip
    elif crop_idx == 8:
        im = im.crop((0, int(h*0.1), w-int(w*0.1), h))
        im = im.transpose(Image.FLIP_LEFT_RIGHT) # bottom left and HFlip
    elif crop_idx == 9:
        im = im.crop((int(w*0.1), int(h*0.1), w, h))
        im = im.transpose(Image.FLIP_LEFT_RIGHT) # bottom right and HFlip
    return im

def predict_without_tta(model, transform, test_data, data_path='./test/0/'):
    """
    function to get the predictions without test time augmentation (this function create csv file for submission)
    --
    INPUTS:
    model: (model object) 
    transform : (transform object)
    test_data : (data frame) containing the category of the image and the path
    data_path : (str) path for the test data
    --
    OUTPUTS: No output
    """ 
    since = time.time()
    
    test_dataset = CassavaTestDataset(test_data, transform=transform, data_path=data_path)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size//2)

    model.eval()
    results = []
    threshold = 4
    print('Inferencing ...')
    for images, image_names in test_loader:
        images = images.to(args.device)
        output = model(images)
        preds = torch.argmax(output, dim=-1)
        preds = preds.cpu().detach().numpy()
       
        for pred, image_name in zip(preds, image_names):
            
            results.append({'Id':image_name, 'Category': classes[pred]})
            

    df = pd.DataFrame(results, columns=['Category', 'Id'])
    df.to_csv('sub.csv', index=False)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def predict_with_tta(model, transform, test_data, num_of_tta=10, data_path='./test/0/'):  
    """
    function to get the predictions using test time augmentation (this function create csv file for submission)
    --
    INPUTS:
    model: (model object) 
    transform : (transform object)
    test_data : (data frame) containing the category of the image and the path
    data_path : (str) path for the test data
    num_of_tta : (int) number of augmentations 
    --
    OUTPUTS: No output
    """ 
    since = time.time()

    assert num_of_tta <= 10, "TTA number must not be more than 10"

    TTA10 = []
    results = []
    print(f'Inferencing with {num_of_tta} TTA ...')

    for tta_idx in range(num_of_tta):
        test_dataset = CassavaTestDataset(test_data, transform=transform, tta=True, tta_idx=tta_idx, data_path=data_path )
        
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size//2, num_workers=0)
        model.eval()

        names2probs = {}
        for images, image_names in test_loader:
            images = images.to(args.device)
            output = model(images)

            probs = F.softmax(output, dim=-1)
            probs = probs.cpu().detach().numpy()
            for prob, image_name in zip(probs, image_names):
                names2probs[image_name] = prob

        TTA10.append(names2probs)

    for im_name in TTA10[0]:
        # Find average of all prediction probabilities for each class
        avg_prob = torch.zeros(5)
        for prob_idx in range(num_of_tta):
            avg_prob += TTA10[prob_idx][im_name]

        avg_prob = avg_prob/num_of_tta
        pred = torch.argmax(avg_prob)
        pred = pred.cpu().detach().numpy()

        results.append({'Id':im_name, 'Category':classes[pred]})

    df = pd.DataFrame(results, columns=['Category', 'Id'])
    df.to_csv('sub_tta.csv', index=False)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))