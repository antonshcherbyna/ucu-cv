import torch
from torch.optim import Adam, SGD

class Trainer:
    '''Class for model training'''

    def __init__(self, model, optimizer, scheduler, criterion, clip_norm,
                 writer, num_updates, device, multi_gpu):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = num_updates

    def train_step(self, batch):
        images, labels = batch

        loss, accuracy = self.forward(images, labels, train=True)
        self.backward(loss)

        return loss, accuracy

    def test_step(self, batch):
        images, labels = batch

        loss, accuracy = self.forward(images, labels, train=False)

        return loss, accuracy

    def forward(self, images, labels, train):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            self.num_updates += 1
        else:
            self.model.eval()

        probs = self.model(images)
        loss = self.criterion(probs, labels)

        _, predicted = torch.max(probs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / images.shape[0]

        return loss, accuracy

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
