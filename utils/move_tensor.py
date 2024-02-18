import torch
import collections

def detach_to_cpu(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        if isinstance(item, collections.abc.Mapping):
            return {k: detach_to_cpu(v) for k, v in item.items()}
        if isinstance(item, collections.abc.Sequence):
            return type(item)(detach_to_cpu(v) for v in item)
        else:
            raise TypeError(f"type {type(item)} can not be moved to cpu")

def transfer_to_device( item, device):
        if isinstance(item, torch.Tensor):
            return item.to(device)

        if isinstance(item, collections.abc.Mapping):
            return {k: transfer_to_device(v, device) for k, v in item.items()}

        if isinstance(item, collections.abc.Sequence):
            return type(item)(transfer_to_device(v, device) for v in item)

        else:
            raise TypeError(f"type {type(item)} can not be moved to device")