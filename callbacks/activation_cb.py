import matplotlib.pyplot as plt
from .base_callback import Callback
import torch


class ActivationCallback(Callback):
    def __init__(self, h_ratio_k=2, width_per_layer=5, height_per_layer=5, n_cols=2):
        super().__init__()

        self.h_ratio_k = h_ratio_k
        self.width_per_layer = width_per_layer
        self.height_per_layer = height_per_layer
        self.n_cols = n_cols

        self.activation_means = []
        self.activation_stds = []
        self.hooks = []

    def before_fit(self):
        self.activation_means = [[] for _ in self.learner.model]
        self.activation_stds = [[] for _ in self.learner.model]
        self.activation_hists = [[] for _ in self.learner.model]
        self.handles = []
        for i, layer in enumerate(self.learner.model):

            def hook_i(module, input, output, layer_i=i):
                # important to use layer_i because functions close over variables, not values
                # Read: https://eev.ee/blog/2011/04/24/gotcha-python-scoping-closures/
                if self.learner.epoch_context["mode"] == "train":
                    self.activation_means[layer_i].append(output.mean().item())
                    self.activation_stds[layer_i].append(output.std().item())
                    self.activation_hists[layer_i].append(output.abs().histc(50, 0, 10))

            self.hooks.append(hook_i)
            layer_handle = layer.register_forward_hook(hook_i)
            self.handles.append(layer_handle)

    def after_fit(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def layer_stats(self, layer_id):
        return (
            self.activation_means[layer_id],
            self.activation_stds[layer_id],
            self.activation_hists[layer_id],
        )

    def plot_layer_activation_lines(self, layer_id, ax=None):
        if ax is not None:
            if len(ax) != 2:
                raise ValueError(
                    "need 2 ax objects to plot mean and standard deviation"
                )
        else:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))

        mean_ax, std_ax = ax
        mean_ax.plot(self.activation_means[layer_id], label=f"Layer {layer_id}")
        std_ax.plot(self.activation_stds[layer_id], label=f"Layer {layer_id}")

        mean_ax.set_ylabel("Means")
        mean_ax.set_xlabel("Batches")

        std_ax.set_ylabel("Standard deviations")
        std_ax.set_xlabel("Batches")
        plt.legend()

    def plot_activation_lines(self, layers=None):
        if layers is None:
            layers = [layer for layer in range(len(self.activation_means))]

        _, ax = plt.subplots(1, 2, figsize=(10, 5))

        mean_ax, std_ax = ax
        for layer in layers:
            self.plot_layer_activation_lines(layer, ax)

    def plot_layer_activation_histogram(self, layer_id, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        histograms = torch.stack(self.activation_hists[layer_id]).t().float().log1p()
        ax.imshow(histograms, cmap="viridis", origin="lower")
        ax.set_xlabel("batches")

    def plot_activation_histograms(self, layers=None):
        if layers is None:
            layers = [layer for layer in range(len(self.activation_hists))]

        n_rows = (len(layers) + self.n_cols - 1) // self.n_cols
        n_cols = self.n_cols
        figsize = n_rows * self.height_per_layer, n_cols * self.width_per_layer
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for layer, ax in zip(layers, ax.flatten()):
            self.plot_layer_activation_histogram(layer, ax)
            ax.set_title(f"Layer {layer}")

        fig.tight_layout()

    def plot_layer_activation_dead_ratio(self, layer_id, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        hists = self.activation_hists[layer_id]
        k = self.h_ratio_k
        h_ratios = [hist[:k].sum() / hist.sum() for hist in hists]
        ax.plot(h_ratios, label=f"Layer {layer_id}")
        ax.set_xlabel("Batches")
        ax.set_ylabel("$dead_{k}$ ratio".format(k=k))
        ax.set_ylim(0, 1.1)

    def plot_activation_dead_ratios(self, layers=None):
        if layers is None:
            layers = [layer for layer in range(len(self.activation_hists))]

        n_rows = (len(layers) + self.n_cols - 1) // self.n_cols
        n_cols = self.n_cols
        figsize = n_rows * self.height_per_layer, n_cols * self.width_per_layer
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)

        for layer, ax in zip(layers, ax.flatten()):
            self.plot_layer_activation_dead_ratio(layer, ax)
            ax.set_title(f"Layer {layer}")

        fig.tight_layout()
