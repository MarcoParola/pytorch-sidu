# **pytorch-sidu**
SIDU: SImilarity Difference and Uniqueness method for explainable AI

- Pytorch implementation of the SIDU method. 
- Simple interface for loading pretrained models by specifying one of the following [string name](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- Clear interface for generating saliency maps

Some examples made with VGG19 on [Caltech-101 dataset](https://paperswithcode.com/dataset/caltech-101):

[img1-1.pdf](https://github.com/MarcoParola/pytorch-sidu/files/14725698/img1-1.pdf)
[img3-1.pdf](https://github.com/MarcoParola/pytorch-sidu/files/14725700/img3-1.pdf)
[img7-1.pdf](https://github.com/MarcoParola/pytorch-sidu/files/14725704/img7-1.pdf)


## Installation

```
pip install pytorch-sidu
```

## Usage

Load models from the pretrainde ones available in pytorch

```py
import pytorch_sidu as sidu

weights = "ResNet18_Weights.IMAGENET1K_V1"
backbone = sidu.load_torch_backbone(weights)
```

After instantianting your model, generate saliency maps from Dataloader

```py
data_loader = <your dataloader>
image, _ = next(iter(data_loader))
saliency_maps = sidu.sidu(backbone, image)
```

### A complete example on CIFAR-10

```py
import torch
import torchvision
from matplotlib import pyplot as plt
import pytorch_sidu as sidu


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)), torchvision.transforms.ToTensor()])
data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform), batch_size=2)

weights = "ResNet18_Weights.IMAGENET1K_V1"
backbone = sidu.load_torch_backbone(weights)

for image, _ in data_loader:
    saliency_maps = sidu.sidu(backbone, image)
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

## License

This project is licensed under the GNU General Public License 3.0. For more details, see the LICENSE file in the root directory of this project or check out the GNU General Public License 3.0. [here](https://www.gnu.org/licenses/gpl-3.0.html)
