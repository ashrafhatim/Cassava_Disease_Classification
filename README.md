# Cassava_Disease_Classification

The final project of the Computer Vision Part 1 course during AMMI master 2021.

![download](https://user-images.githubusercontent.com/45710249/135855772-bd28cd86-4a16-4ce8-8fc8-e0315d94ab82.jpeg)

### Objective :

The objective is to provide a model to classify images of the cassava plant
into 4 disease categories and healthy, given 9,436 annotated images and 12,595
unlabeled images of cassava leaves. We tackled the problem with two phases, first we used the labeled images to find
our best model, then we fine-tune our best model using both the labeled and
unlabeled images.

### To download the requirements :
```pip install -r requirements.txt```


### Note:

The script contains only the main experiment , please look at the other experiments in the notebook folder.

### Results :

Model| Train Accuracy| Evaluation Accuracy |Public Leader-board |
------------ | -------------|---------------|-------------------|
RESNET50| 86.3 |   87.1     |      88.01    |              
RESNEXT50| 87.4|   87.9   |    88.3     |
RESNECT101| 90.1|   90.5   |     90.26    |
PSEUDO-LABELING|91.3| 92.3  |   90.5 |


### Conclusion :
```RESNEXT101 model with semi-supervised learning achieved the best result.```
