# pytorch_sidu/sidu.py

r""" SIDU SImilarity Difference and Uniqueness method

The SIDU method is a method for explaining the predictions of a model by computing the similarity differences and the uniqueness of the masks generated from the feature map of the model. The method is based on the paper **SIDU: Similarity Difference and Uniqueness Method for Explainable AI** by Satya M. Muddamsetty, Mohammad N. S. Jahromi, and Thomas B. Moeslund.

The module contains the following functions:
    - kernel: Kernel function for computing the weights of the differences
    - normalize: Normalize the array
    - uniqness_measure: Compute the uniqueness measure
    - similarity_differences: Compute the similarity differences
    - generate_masks: Generate masks from the feature map
    - sidu: SIDU SImilarity Difference and Uniqueness method
"""

import torch
import torchvision
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor


def kernel(vector: torch.Tensor, kernel_width: float = 0.1) -> torch.Tensor:
    """
    Kernel function for computing the weights of the differences.

    Args:
        vector (torch.Tensor): 
            The difference tensor.
        kernel_width (float, optional): 
            The kernel width. Defaults to 0.1.

    Returns:
        torch.Tensor: 
            The weights.
    """
    return torch.sqrt(torch.exp(-(vector ** 2) / kernel_width ** 2))


def normalize(array: torch.Tensor) -> torch.Tensor:
    r"""
    Normalize the array

    Args:
        array: torch.Tensor
            The input array

    Returns:
        normalized_array: torch.Tensor
            The normalized array
    """
    return (array - array.min()) / (array.max() - array.min() + 1e-13)



def uniqness_measure(masked_feature_map: torch.Tensor) -> torch.Tensor:
    r""" Compute the uniqueness measure

    Args:
        masked_feature_map: torch.Tensor
            The masked feature map

    Returns:
        uniqueness: torch.Tensor
            The uniqueness measure
    """
    batch_size, num_masks, feature_len, w, h = masked_feature_map.size()

    # Reshape tensor to have size (batch * num_masks, feature_len, w * h)
    tensor_reshaped = masked_feature_map.view(batch_size * num_masks, feature_len, -1)

    # Compute pairwise distances between each feature vector
    distances = torch.cdist(tensor_reshaped, tensor_reshaped, p=2.0)
    distances = distances.view(batch_size, num_masks, num_masks, -1)

    # Compute mean along the last tow dimensions to get uniqueness measure for each mask
    uniqueness = distances.mean(dim=-1).mean(dim=-1)

    return uniqueness



def similarity_differences(orig_predictions: torch.Tensor, masked_predictions: torch.Tensor):
    r""" Compute the similarity differences

    Args:
        orig_predictions: torch.Tensor
            The original predictions
        masked_predictions: torch.Tensor
            The masked predictions

    Returns:
         : torch.Tensor
            The weights
        diff: torch.Tensor
            The differences
    """
    diff = abs(masked_predictions - orig_predictions)
    # compute the average of the differences, from (batch, num_masks, num_features, w, h) -> (batch, num_masks, w, h)
    diff = diff.mean(dim=2)
    weighted_diff = kernel(diff)
    # compute the average of the weights, from (batch, num_masks, w, h) -> (batch, num_masks)
    weighted_diff = weighted_diff.mean(dim=(2, 3))
    return weighted_diff, diff



def generate_masks(img_size: tuple, feature_map: torch.Tensor, s: int = 8) -> torch.Tensor:
    r""" Generate masks from the feature map

    Args:
        img_size: tuple
            The size of the input image
        feature_map: torch.Tensor
            The feature map from the model
        s: int
            The scale factor

    Returns:
        masks: torch.Tensor
            The generated masks
    """
    h, w = img_size
    cell_size = np.ceil(np.array(img_size) / s)
    up_size = (s) * cell_size
    N = feature_map.shape[1]
    batch = feature_map.shape[0]
    
    # converting to binary mask then to float
    feature_map = (feature_map > 0.15).float()
    
    # resize to the size of the input image torch.nn.functional.F.interpolate: (batch, N, f_h, f_w) -> (batch, N, h, w)
    masks = torch.nn.functional.interpolate(feature_map, size=(h, w), mode='bilinear', align_corners=True)
        
    return masks


def sidu(model: torch.nn.Module, layer_name: str, image: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    r""" SIDU SImilarity Difference and Uniqueness method
    
    Args:
        model: torch.nn.Module
            The model to be explained
        image: torch.Tensor
            The input image to be explained
        layer_name: str
            The layer name of the model to be explained
            It must be contained in named_modules() of the model
        device: torch.device, optional
            The device to use. Defaults to torch.device("cpu")

    Returns:
        saliency_maps: torch.Tensor
            The saliency maps

    """
    model.eval()
    model.to(device)
    image = image.to(device)
    return_nodes = {layer_name: 'target_layer'}
    model = create_feature_extractor(model, return_nodes=return_nodes)
    
    with torch.no_grad():
        orig_feature_map = model(image)['target_layer']
        input_size = image.shape[2:]
        masks = generate_masks(input_size, orig_feature_map, 16)
        batch = image.shape[0]
        num_masks = masks.shape[1]

        # Repeat the masks 3 times along the channel dimension to match the number of channels of the image and masks
        masks = masks.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        images = image.unsqueeze(1).repeat(1, num_masks, 1, 1, 1)
        masked_images = images * masks

        masked_feature_map = []
        for i in range(num_masks):
            masked_feature_map.append(model(masked_images[:, i, :, :, :])['target_layer'])
        masked_feature_map = torch.stack(masked_feature_map, dim=1) # TODO speed up this part
        
        orig_feature_map = orig_feature_map.unsqueeze(1).repeat(1, num_masks, 1, 1, 1)

        # compute the differences of the similarity and the uniqueness
        weighted_diff, difference = similarity_differences(orig_feature_map, masked_feature_map)
        uniqness = uniqness_measure(masked_feature_map)

        sidu = weighted_diff * uniqness


        # reduce the masks size by removing the channel dimension (batch, num_masks, 3, w, h) -> (batch, num_masks, 1, w, h)
        masks = masks.mean(dim=2, keepdim=True)
        masks = masks.squeeze(2)

        # compute saliency maps by averaging the masks weighted by the SIDU
        # each mask of masks (batch, num_masks, w, h) must be multiplied by the SIDU (batch, num_masks)
        saliency_maps = masks * sidu.unsqueeze(2).unsqueeze(3)

        # reduce the saliency maps to a single map by summing over the masks dimension
        saliency_maps = saliency_maps.sum(dim=1)
        saliency_maps /= num_masks

        return saliency_maps
