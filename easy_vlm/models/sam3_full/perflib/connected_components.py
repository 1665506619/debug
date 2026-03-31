import logging

import torch


try:
    from cc_torch import get_connected_components

    HAS_CC_TORCH = True
except ImportError:
    logging.debug("cc_torch not found. Falling back to CPU connected components.")
    HAS_CC_TORCH = False


def connected_components_cpu_single(values: torch.Tensor):
    assert values.dim() == 2
    from skimage.measure import label

    labels, num = label(values.cpu().numpy(), return_num=True)
    labels = torch.from_numpy(labels)
    counts = torch.zeros_like(labels)
    for i in range(1, num + 1):
        cur_mask = labels == i
        counts[cur_mask] = cur_mask.sum()
    return labels, counts


def connected_components_cpu(input_tensor: torch.Tensor):
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert input_tensor.dim() == 3, "Input tensor must be (B, H, W) or (B, 1, H, W)."

    if input_tensor.shape[0] == 0:
        empty = torch.zeros(out_shape, device=input_tensor.device, dtype=torch.int64)
        return empty, empty.clone()

    labels_list = []
    counts_list = []
    for b in range(input_tensor.shape[0]):
        labels, counts = connected_components_cpu_single(input_tensor[b])
        labels_list.append(labels)
        counts_list.append(counts)
    labels_tensor = torch.stack(labels_list, dim=0).to(input_tensor.device)
    counts_tensor = torch.stack(counts_list, dim=0).to(input_tensor.device)
    return labels_tensor.view(out_shape), counts_tensor.view(out_shape)


def connected_components(input_tensor: torch.Tensor):
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)
    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1, (
        "Input tensor must be (B, H, W) or (B, 1, H, W)."
    )

    if input_tensor.is_cuda and HAS_CC_TORCH:
        return get_connected_components(input_tensor.to(torch.uint8))
    return connected_components_cpu(input_tensor)
