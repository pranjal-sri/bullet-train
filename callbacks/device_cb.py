import collections
import torch
from .base_callback import Callback
from ..utils.move_tensor import transfer_to_device

class DeviceCallback(Callback):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def before_fit(self):
        self.learner.model.to(self.device)


    def before_batch(self):
        self.learner.batch_context["x"] = transfer_to_device(
            self.learner.batch_context["x"], self.device
        )
        self.learner.batch_context["y"] = transfer_to_device(
            self.learner.batch_context["y"], self.device
        )
