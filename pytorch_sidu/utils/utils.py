import torch
import torchvision

def load_torch_backbone(model_name, layer=-2):
    r""" Load a backbone model from torchvision
    Args:
        model_name: str
            The name of the model
    Returns:
        backbone: torch.nn.Sequential
            The backbone model
    """
    name, version = model_name.split(".")
    name = name.split("_Weights")[0].lower()
    model = getattr(torchvision.models, name)(weights=model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    
    backbone = torch.nn.Sequential(*list(model.children())[:layer])
    return backbone