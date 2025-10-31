import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    """y_pred is expected to be label indices, not logits."""
    correct = (y_true == y_pred).sum().item()
    total = len(y_true)
    return (correct / total) * 100


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
class_names = train_data.classes


BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=False)
train_features_batch, train_labels_batch = next(iter(train_dataloader))
flatten_model = nn.Flatten() 
x = train_features_batch[0]

output = flatten_model(x) 
loss_fn = nn.CrossEntropyLoss()


def eval_model(model, dataloader, loss_fn, accuracy_fn):
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      y_pred = model(X)
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y, y_pred.argmax(dim=1))
    loss /= len(dataloader)
    acc /= len(dataloader)

  return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

def train_step(model, dataloader, loss_fn, optimizer, accuracy_fn):
  train_loss, train_acc = 0, 0
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def eval_step(model, dataloader, loss_fn, accuracy_fn):
  test_loss, test_acc = 0, 0
  model.eval()

  with torch.inference_mode():
    for X, y in dataloader:
      test_pred = model(X)
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    
class FashionClassifierV2(nn.Module):
  def __init__(self, input_shape, hidden, output_shape):
    super().__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(input_shape, hidden, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block2 = nn.Sequential (
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    self.classifier = nn.Sequential (
        nn.Flatten(),
        nn.Linear(hidden*7*7, output_shape)
    )

  def forward(self, X):
    X = self.block1(X)
    X = self.block2(X)
    X = self.classifier(X)
    return X

model_2 = FashionClassifierV2(1, 10, len(class_names))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.1)

epochs = 10
torch.manual_seed(42)

for epoch in range(epochs):
  print(f"Epoch: {epoch}\n---------")
  train_step(model_2, train_dataloader, loss_fn, optimizer, accuracy_fn)
  eval_step(model_2, test_dataloader, loss_fn, accuracy_fn)

model2_results = eval_model(model_2, test_dataloader, loss_fn, accuracy_fn)
print(model2_results)