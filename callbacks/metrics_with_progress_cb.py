from fastprogress import master_bar, progress_bar
import fastcore.foundation as fc
from .metrics_cb import MetricsCallback


class MetricsWithProgressCallback(MetricsCallback):
    def __init__(self, *unnamed_metrics, plot=False, **named_metrics):
        super().__init__(*unnamed_metrics, **named_metrics)
        self.plot = plot

    def before_fit(self):
        super().before_fit()
        self.mbar = master_bar(self.learner.fit_context["epochs"])
        self.learner.fit_context["epochs"] = self.mbar
        self.losses = []

    #  Called by super class after epoch
    def _log(self, log):
        self.mbar.write(f'{log}')

    def before_epoch(self):
        super().before_epoch()
        self.learner.epoch_context["dl"] = progress_bar(
            self.learner.epoch_context["dl"], leave = False, parent=self.mbar
        )

    def after_batch(self):
        super().after_batch()
        self.mbar.child.comment = f'{self.learner.batch_op["loss"]:.3f}'
        if self.plot and self.learner.epoch_context["mode"] == "train":
            self.losses.append(self.learner.batch_op["loss"].item())
            self.mbar.update_graph([(fc.L.range(self.losses), self.losses)])
