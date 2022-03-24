import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pytorch_lightning as pl


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Create data loader
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)


class NeuralNetworkLit(pl.LightningModule):
    def __init__(self):
        super(NeuralNetworkLit, self).__init__()

        # loss function
        self.loss_fn =  nn.CrossEntropyLoss()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

# train model
model = NeuralNetworkLit()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_dataloader)
