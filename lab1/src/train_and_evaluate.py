from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

class MyTrainerEvaluator():
    def __init__(self, model, optimizer, loss_function, logger, device, epochs, train_dl, val_dl, test_dl):
        self.model = model.to(device)
        self.optim = optimizer
        self.loss_function = loss_function
        self.logger = logger
        self.device = device
        self.epochs = epochs
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl


    def train_epoch(self, epoch='Unknown'):
        self.model.train()
        losses = []
        for (xs, ys) in tqdm(self.train_dl, desc=f'Training epoch {epoch}', leave=True):
            xs = xs.to(self.device)
            ys = ys.to(self.device)
            self.optim.zero_grad()
            logits = self.model(xs)
            loss = self.loss_function(logits, ys)
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        return np.mean(losses)

    def evaluate_model(self, validation):

        if validation:
            dl = self.val_dl
            print('Validation set will be used!')
        else:
            dl = self.test_dl
            print('Test set will be used!')

        self.model.eval()
        predictions, gts = [], []
        for (xs, ys) in tqdm(dl, desc='Evaluating', leave=False):
            xs = xs.to(self.device)
            preds = torch.argmax(self.model(xs), dim=1)
            gts.append(ys)
            predictions.append(preds.detach().cpu().numpy())
        
        return (accuracy_score(np.hstack(gts), np.hstack(predictions)),
            classification_report(np.hstack(gts), np.hstack(predictions), zero_division=0, digits=3))
    

    def train_and_evaluate(self):
        losses = []
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch=epoch+1)
            losses.append(train_loss)
            (val_accuracy, _) = self.evaluate_model(validation=True)

            print(f'Epoch {epoch}: Training Loss: {train_loss}  Validation accuracy: {val_accuracy}')

        plt.plot(losses)
        (_, test_cls_report) = self.evaluate_model(validation=False)
        print(test_cls_report)