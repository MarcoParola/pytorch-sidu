# **pytorch-sidu**

[![size](https://img.shields.io/github/languages/code-size/MarcoParola/pytorch-sidu)]()

SIDU: SImilarity Difference and Uniqueness method for explainable AI from the [original paper](https://arxiv.org/pdf/2006.03122.pdf)

- Pytorch implementation of the SIDU method. 
- Simple interface for loading pretrained models by specifying one of the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- Clear interface for generating saliency maps

Some examples made with VGG19 on [Caltech-101 dataset](https://paperswithcode.com/dataset/caltech-101):

![img1](https://github.com/MarcoParola/pytorch-sidu/assets/32603898/e2bc0085-11c8-4fd7-975e-72e49ff7ee77)
![img7](https://github.com/MarcoParola/pytorch-sidu/assets/32603898/860492cf-fc24-4f40-ad65-6d42a6a539a8)
![img9](https://github.com/MarcoParola/pytorch-sidu/assets/32603898/83c7c206-5927-438d-93af-aa3e94914461)


## Installation

```
pip install pytorch-sidu
```

## Usage

Load models from the pretrainde ones available in pytorch

```py
import pytorch_sidu as sidu

model_name = 'ResNet34_Weights.IMAGENET1K_V1'
model = load_torch_model_by_string(model_name)
```

After instantianting your model, generate saliency maps from Dataloader

```py
data_loader = <your dataloader>
target_layer = 'layer4.2.conv2'
image, _ = next(iter(data_loader))
saliency_maps = sidu.sidu(model, target_layer, image)
```

### A complete example on CIFAR-10

```py
import torch
import torchvision
from matplotlib import pyplot as plt
import pytorch_sidu as sidu


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform), batch_size=2)

target_layer = 'layer4.2.conv2'
model_name = 'ResNet34_Weights.IMAGENET1K_V1'
model = load_torch_model_by_string(model_name)

for image, _ in data_loader:
    saliency_maps = sidu.sidu(model, target_layer, image)
    image, saliency_maps = image.cpu(), saliency_maps.cpu()

    for j in range(len(image)):
        plt.figure(figsize=(5, 2.5))
        plt.subplot(1, 2, 1)
        plt.imshow(image[j].permute(1, 2, 0))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image[j].permute(1, 2, 0))
        plt.imshow(saliency_maps[j].squeeze().detach().numpy(), cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.show()
```

## upcoming features:
- integration of xai metrics
- make methods work on both single images and dataloaders
- adding `device` flag to sidu function to allow device selection
