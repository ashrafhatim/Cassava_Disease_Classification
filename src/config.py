import argparse
import torch

args = argparse.Namespace(
    size = 448,
    # Model Hyperparameters
    learning_rate = 2e-4,
    batch_size = 4,
    num_epochs = 20,
    valid_num_epochs = 10,
    momentum=0.9,

    # Data Parameters
    mean = torch.tensor([0.485, 0.456, 0.406]),
    std = torch.tensor([0.229, 0.224, 0.225]),
    validation_split = .1,
    shuffle_dataset = True,
    num_folds=5,

    seed= 0,

    # Paths
    save_dir =  "./checkpoints", #"/content/drive/MyDrive/kaggle/cassave/",
    train_path = "./data/train",
    test_path = "./data/test",


    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # semisup args
    T1 = 100,
    T2 = 700,
    af = 3,

)