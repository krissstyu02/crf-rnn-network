import torch
import torch.nn as nn
import torch.nn.functional as F



class CRF_RNN(nn.Module):
    def __init__(self, num_classes):
        super(CRF_RNN, self).__init__()

        # Define the CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the CRF-RNN
        self.crf_rnn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
            # Run input through CNN backbone
        x = self.cnn(x)

            # Run CNN output through CRF-RNN
        x = self.crf_rnn(x)

        x = x.mean(dim=2).mean(dim=2)

        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)

        return x


import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

# Define transformations to be applied to the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
val_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

# Define the model
model = CRF_RNN(num_classes=10)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     print("Epoh 1")
#     for i, data in enumerate(train_loader, 0):
#         # Get the inputs and labels
#         inputs, labels = data
#
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#
#         # Forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # Print statistics
#         running_loss += loss.item()
#         if i % 100 == 99:  # Print every 100 mini-batches
#             print('[Epoch %d, Batch %5d] Loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0
#
# print('Finished training')



import matplotlib.pyplot as plt

# Select an image from the test set
image, label = test_dataset[1000]



# Pass the image through the model to obtain segmentation mask
with torch.no_grad():
    output = model(image.unsqueeze(0))


# Convert the output to a numpy array
output = output.squeeze(0).argmax(dim=0).cpu().numpy()

# Display the original image and segmentation mask
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

ax1.imshow(image.permute(1, 2, 0).numpy())
ax1.set_title('Original Image')


# ax2.imshow(output)
# ax2.set_title('Segmentation Mask')

plt.show()


#обучить модель,
# разобраться почему не выводится сегментация,
# почему картинки в плохом качестве