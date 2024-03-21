import torch
import torchvision
import numpy as np



def kernel(d, kernel_width=0.25):
    r""" Kernel function for the similarity differences
    Args:
        d: torch.Tensor
            The difference tensor
        kernel_width: float
            The kernel width
    Returns:
        weights: torch.Tensor
            The weights
    """
    #return torch.exp(-(d ** 2) / kernel_width ** 2)
    return torch.sqrt(torch.exp(-(d ** 2) / kernel_width ** 2))

def normalize(array):
      return (array - array.min()) / (array.max() - array.min() + 1e-13)



def uniqness_measure(masked_feature_map):
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



def similarity_differences(original_predictions, masked_predictions):
    r""" Compute the similarity differences
    Args:
        original_predictions: torch.Tensor
            The original predictions
        masked_predictions: torch.Tensor
            The masked predictions
    Returns:
         : torch.Tensor
            The weights
        diff: torch.Tensor
            The differences
    """
    diff = abs(masked_predictions - original_predictions)
    # compute the average of the differences, from (batch, num_masks, num_features, w, h) -> (batch, num_masks, w, h)
    diff = diff.mean(dim=2)
    weighted_diff = kernel(diff)
    # compute the average of the weights, from (batch, num_masks, w, h) -> (batch, num_masks)
    weighted_diff = weighted_diff.mean(dim=(2, 3))
    return weighted_diff, diff



def generate_masks(img_size, feature_map, s=8):
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



def sidu(model, image):
    r""" SIDU SImilarity Difference and Uniqueness method
    Args:
        model: torch.nn.Module
            The model to be explained
        image: torch.Tensor
            The input image to be explained
        masks: torch.Tensor
            The generated masks
    """

    model.eval()
    with torch.no_grad():
        orig_feature_map = model(image)
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
            masked_feature_map.append(model(masked_images[:, i, :, :, :]))
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
        # each tensor of w x h must be multiplied by the corresponding SIDU value
        saliency_maps = masks * sidu.unsqueeze(2).unsqueeze(3)

        saliency_maps_weighted_diff_only = masks * weighted_diff.unsqueeze(2).unsqueeze(3)
        saliency_maps_uniqness_only = masks * uniqness.unsqueeze(2).unsqueeze(3)

        # reduce the saliency maps to a single map by summing over the masks dimension
        saliency_maps = saliency_maps.sum(dim=1)
        saliency_maps /= num_masks
        
        saliency_maps_weighted_diff_only = saliency_maps_weighted_diff_only.sum(dim=1)
        saliency_maps_weighted_diff_only /= num_masks

        saliency_maps_uniqness_only = saliency_maps_uniqness_only.sum(dim=1)
        saliency_maps_uniqness_only /= num_masks

        return saliency_maps, saliency_maps_weighted_diff_only, saliency_maps_uniqness_only




def load_torch_backbone(model_name, layer=-4):
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
