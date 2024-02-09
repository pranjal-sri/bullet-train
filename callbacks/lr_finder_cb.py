import math
from torch.optim.lr_scheduler import ExponentialLR
from .base_callback import Callback
from ..learners.exceptions import CancelEpochException, CancelFitException
import matplotlib.pyplot as plt

class LRFinderCallback(Callback):
    def __init__(self, gamma = 1.3, stop_factor = 5):
        super().__init__()
        self.gamma = gamma
        self.stop_factor = 3

    def before_fit(self):
        self.sched = ExponentialLR(self.learner.optimizer, self.gamma)
        self.losses = []
        self.learning_rates = []
        self.min = float('inf')

    def before_epoch(self):
        if self.learner.epoch_context['mode'] == 'evaluate':
            raise CancelEpochException
        
    def after_batch(self):
        self.learning_rates.append(self.learner.optimizer.param_groups[0]['lr'])
        loss = self.learner.batch_op['loss'].detach().cpu()
        self.losses.append(loss)
        if loss < self.min:
            self.min = loss
        if math.isnan(loss) or (loss > self.min*self.stop_factor):
            raise CancelFitException()
        self.sched.step()

    def plot(self):
      plt.plot(self.learning_rates, self.losses)
      plt.xscale('log')
      plt.xlabel('Learning rates')
      plt.ylabel('Loss')
      plt.title('Loss vs Learning rate')
      plt.show()
