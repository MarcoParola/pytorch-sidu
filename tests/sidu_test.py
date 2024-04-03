import torch
import torchvision
import pytorch_sidu as sidu


def test_sidu():
    model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    batch_size = 1
    w = 256
    h = 256
    layer_name = 'layer4.1.conv2'

    image = torch.rand(batch_size, 3, w, h)
    saliency_map = sidu.sidu(model, layer_name, image, device=torch.device('cuda'))
    assert saliency_map.shape == (batch_size, w, h)

