import collections
import torch
from .base_callback import Callback


class DeviceCallback(Callback):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def before_fit(self):
        self.learner.model.to(self.device)

    def __transfer_to_device(self, item):
        if isinstance(item, torch.Tensor):
            return item.to(self.device)

        if isinstance(item, collections.abc.Mapping):
            return {k: self.transfer_to_device(v) for k, v in item.items()}

        if isinstance(item, collections.abc.Sequence):
            return type(item)(self.transfer_to_device(v) for v in item)

        else:
            raise TypeError(f"type {type(item)} can not be moved to device")

    def before_batch(self):
        self.learner.batch_context["x"] = self.__transfer_to_device(
            self.learner.batch_context["x"]
        )
        self.learner.batch_context["y"] = self.__transfer_to_device(
            self.learner.batch_context["y"]
        )
