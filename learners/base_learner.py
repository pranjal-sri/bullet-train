import fastcore.foundation as fc
from contextlib import contextmanager
import torch

from .exceptions import CancelBatchException, CancelEpochException, CancelFitException
from ..callbacks import Callback, DeviceCallback, MetricsWithProgressCallback


class AbstractLearner:
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_dl,
        validation_dl,
        callbacks=None,
        load_default_callbacks=True,
        plot=True,
        device=None,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.train_dl = train_dl
        self.validation_dl = validation_dl

        self.__callbacks = []

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.plot = plot

        self.batch_op = {}
        self.epoch_context = {}
        self.batch_context = {}
        self.fit_context = {}

        self.__init_setup_callbacks(callbacks, load_default_callbacks, plot)

    def predict(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def register_callback(self, callback):
        if not isinstance(callback, Callback):
            raise TypeError(f"{callback} is not a Callback")

        if callback in self.__callbacks:
            raise ValueError(f"{callback} is already registered")

        self.__callbacks.append(callback)
        callback.learner = self

        self.__callbacks = sorted(self.__callbacks, key=lambda cb: cb.order)

    def unregister_callback(self, callback):
        if callback not in self.__callbacks:
            raise ValueError("Callback does not exist, unregister attempt failed")

        self.__callbacks.remove(callback)
        callback.learner = None

    def __init_setup_callbacks(self, callbacks, load_default_callbacks, plot):
        for cb in fc.L(callbacks):
            self.register_callback(cb)

        if load_default_callbacks:
            self.device = DeviceCallback(self.device)
            self.register_callback(self.device)
            
            self.metrics =MetricsWithProgressCallback(plot = plot)
            self.register_callback(self.metrics)

    def __run_callbacks(self, function_name):
        for callback in self.__callbacks:
            method = getattr(callback, function_name, None)
            if method is not None:
                method()

    @contextmanager
    def __callback_context(self, step):
        try:
            self.__run_callbacks(f"before_{step}")
            yield
            self.__run_callbacks(f"after_{step}")

        # Catches CancelFitException, CancelEpochException, CancelBatchException
        except globals()[f"Cancel{step.title()}Exception"]:
            pass

    def epoch(self):
        try:
            with self.__callback_context("epoch"):
                for batch_number, (xb, yb) in enumerate(self.epoch_context["dl"]):
                    self.batch_context["current_batch"] = batch_number
                    self.batch_context["x"] = xb
                    self.batch_context["y"] = yb
                    self.batch()
        
        finally:
            self.batch_context.clear()

    def batch(self):
        try:
            with self.__callback_context("batch"):
                self.batch_op["predictions"] = self.predict()
                self.batch_op["loss"] = self.loss()
                if self.epoch_context["mode"] == "train":
                    self.backward()
                    self.step()
                    self.zero_grad()
        finally:
            self.batch_op.clear()

    def fit(self, n_epochs, with_validation=True, fit_callbacks=None):
        if fit_callbacks is not None:
            for callback in fc.L(fit_callbacks):
                self.register_callback(callback)
        try:
            self.fit_context["n_epochs"] = n_epochs
            self.fit_context["with_validation"] = with_validation
            self.fit_context["epochs"] = range(1, n_epochs + 1)
            with self.__callback_context("fit"):
                for epoch in self.fit_context["epochs"]:
                    self.epoch_context["current_epoch"] = epoch

                    self.epoch_context["mode"] = "train"
                    self.epoch_context["dl"] = self.train_dl
                    self.model.train(True)

                    self.epoch()

                    if with_validation:
                        self.epoch_context["mode"] = "evaluate"
                        self.epoch_context["dl"] = self.validation_dl
                        self.model.train(False)

                        with torch.no_grad():
                            self.epoch()
        finally:
            self.epoch_context.clear()
            self.fit_context.clear()

            if fit_callbacks is not None:
                for callback in fc.L(fit_callbacks):
                    self.unregister_callback(callback)
