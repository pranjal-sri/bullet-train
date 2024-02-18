from copy import copy
import collections

import torch
from torcheval.metrics import Mean

from .base_callback import Callback
from ..utils.move_tensor import detach_to_cpu


class MetricsCallback(Callback):
    def __init__(self, *unnamed_metrics, **named_metrics):
        super().__init__()

        for unnamed_metric in unnamed_metrics:
            named_metrics[type(unnamed_metric).__name__] = unnamed_metric

        self.metrics = copy(named_metrics)
        self.loss = Mean()

    # def _transfer_to_cpu(self, item):
    #     if isinstance(item, torch.Tensor):
    #         return item.detach().cpu()
    #     if isinstance(item, collections.abc.Mapping):
    #         return {k: transfer_to_cpu(v) for k, v in item.items()}
    #     if isinstance(item, collections.abc.Sequence):
    #         return type(item)(transfer_to_cpu(v) for v in item)
    #     else:
    #         raise TypeError(f"type {type(item)} can not be moved to cpu")

    def before_fit(self):
        self.learner.fit_context["metrics"] = self.metrics

    def before_epoch(self):
        self.loss.reset()
        for metric in self.metrics.values():
            metric.reset()

    def after_epoch(self):
        log = {
            metric_name: f"{metric.compute():.3f}"
            for metric_name, metric in self.metrics.items()
        }
        log["loss"] = f"{self.loss.compute():.3f}"
        log["epoch"] = self.learner.epoch_context["current_epoch"]
        log["mode"] = self.learner.epoch_context["mode"]

        self._log(log)

    def after_batch(self):
        # import pdb; pdb.set_trace()
        y_true = detach_to_cpu(self.learner.batch_context["y"])
        y_pred = detach_to_cpu(self.learner.batch_op["predictions"])
        batch_loss = detach_to_cpu(self.learner.batch_op["loss"])
        self.loss.update(batch_loss, weight=len(y_true))
        for metric in self.metrics.values():
            metric.update(y_pred, y_true)

    def add_metrics(self, *unnamed_metrics, **named_metrics ):
      for unnamed_metric in unnamed_metrics:
        named_metrics[type(unnamed_metric).__name__] = unnamed_metric
      for name, metric in named_metrics.items():
        self.metrics[name] = metric
      
    def _log(self, log):
        print(log)
