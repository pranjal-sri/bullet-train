from .base_learner import AbstractLearner

class GeneralLearner(AbstractLearner):
    def __init__(self, model, optim, loss_func, train_dl, validation_dl, *args, **kwargs):
        super().__init__(model, optim, loss_func, train_dl, validation_dl, *args, **kwargs)
        self.model = model
        self.optim = optim
        self.loss_func = loss_func

    def predict(self):
        return self.model(self.batch_context['x'])        
    
    def loss(self):
        return self.loss_func(self.batch_op['predictions'], self.batch_context['y'])
    
    def step(self):
        self.optim.step()
    
    def zero_grad(self):
        self.optim.zero_grad()

    def backward(self):
        self.batch_op['loss'].backward()