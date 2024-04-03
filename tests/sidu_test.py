import torch
import torchvision
import pytorch_sidu as sidu
import pytest


@pytest.mark.parametrize("model_name, layer_name", [
    ('ResNet18_Weights.IMAGENET1K_V1', 'layer4.1.conv2'),
    ('ResNet34_Weights.IMAGENET1K_V1', 'layer4.2.conv2'),
    ('ViT_B_16_Weights.IMAGENET1K_V1', 'conv_proj'),
    ('ViT_B_32_Weights.IMAGENET1K_V1', 'conv_proj'),
])
def test_sidu(layer_name, model_name):

    batch_size = 1
    w = 224
    h = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    name, version = model_name.split(".")
    name = name.split("_Weights")[0].lower()
    model = getattr(torchvision.models, name)(weights=model_name)
    model.eval()

    image = torch.rand(batch_size, 3, w, h)
    saliency_map = sidu.sidu(model, layer_name, image, device=device)
    assert saliency_map.shape == (batch_size, w, h)

