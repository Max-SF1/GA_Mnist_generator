import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HyperParams
input_size = 784  # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.005

# MNIST  - LOADING DATASET
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
dataiter = iter(train_loader)
samples, labels = next(dataiter)

#print(samples.shape,
      #labels.shape)  # 100 in a batch, 1 channel, of 28x28 so outputs torch.size([100,1,28,28]) torch.size([100])


class RobustConvNet(nn.Module):
    def __init__(self):
        super(RobustConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # nx1x28x28 -> nx6x24x24
        self.pool = nn.MaxPool2d(2, 2)  # nx6x24X24 -> nx6x12X12
        self.conv2 = nn.Conv2d(6, 14, 3)  # nx14x10x10 -> nx14x5x5 we'll do pool twice.
        self.fc1 = nn.Linear(14 * 5 * 5, 150)
        self.fc2 = nn.Linear(150, 10)

    def forward(self, x):
        out = torch.tanh(self.conv1(x))
        out = torch.relu(self.pool(out))
        out = torch.tanh(self.pool(self.conv2(out)))
        out = out.view(-1, 14 * 5 * 5)  # reshaping into a column vector for the FC layers
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


model = RobustConvNet()
summary(model, (1,28,28))
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images , labels) in enumerate(train_loader):  # index and data
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()  # EMPTYING GRADS
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item()}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        labels = labels.to(device)
        outputs = model(images) #might want to change to image>125 because conv's input is a grayscale image and the GA is binary.
        _, predictions = torch.max(outputs, 1)  # value (don't care), index <- index gives us the class
        n_samples += labels.shape[0]
        n_correct += (
                predictions == labels).sum().item()  # for each prediction check if it equals the label, if it is sum it

acc = 100 * n_correct / n_samples
print("accuracy =" + str(acc))

# Save the trained model
torch.save(model.state_dict(), 'robust_conv_net.pth')
