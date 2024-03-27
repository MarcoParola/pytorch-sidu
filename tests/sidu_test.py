import torch
import torchvision
import pytorch_sidu as sidu


def test_sidu():
    weights = "ResNet18_Weights.IMAGENET1K_V1"
    model = sidu.load_torch_backbone(weights)

    batch_size = 1
    w = 256
    h = 256

    image = torch.rand(batch_size, 3, w, h)
    saliency_map = sidu.sidu(model, image)
    assert saliency_map.shape == (batch_size, w, h)

