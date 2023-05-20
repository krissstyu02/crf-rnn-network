import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm


class CRF_RNN(nn.Module):
    def __init__(self, num_classes):
        super(CRF_RNN, self).__init__()
        self.num_classes = num_classes

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
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
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
            nn.Conv2d(512, 512, kernel_size=3, padding=16, dilation=16),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
            x = self.cnn(x)
            x = self.crf_rnn(x)
            return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

transform = transforms.Compose([
    transforms.ToTensor()  # convert image to tensor
])

def custom_collate(batch):
    images = []
    targets = []
    for sample in batch:
        image, target = sample
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        if not isinstance(target, torch.Tensor):
            target = transforms.ToTensor()(target)
        transform = transforms.Resize((320, 480))
        image = transform(image)
        target = transform(target)
        images.append(image)
        targets.append(target)
    return torch.stack(images), torch.stack(targets)


train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=False, transform=transform)
test_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=transform)


# Define the dataloader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

# Define the model
model = CRF_RNN(num_classes=21)
checkpoint = torch.load('model_55EPOH.pt')
model.load_state_dict(checkpoint)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for images, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        targets = targets.view(-1)
        outputs=outputs.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


num_epochs = 5
# for epoch in range(num_epochs):
#     train_loss = train(model, train_loader, criterion, optimizer)
#     print(f'Epoch {epoch + 1} - Train loss: {train_loss:.4f}')
#
#
# torch.save(model.state_dict(), 'model_55EPOH.pt')
# Define the device to run the model on

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Select the desired image index
index = 100  # Note that indexing starts from 0

# Get the image and target at the specified index
image, target = test_dataset[index]

# Create a mini-batch with a single image
images = image.unsqueeze(0)

# Move the mini-batch to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = images.to(device)

# Get the model predictions
with torch.no_grad():
    outputs = model(images)

# Convert the predictions to a segmentation map
_, predicted = torch.max(outputs.data, 1)
segmentation = predicted.cpu().numpy()[0]

# Get the original image
image = images.cpu().numpy()[0].transpose(1, 2, 0)

import numpy as np
# Get the original segmentation mask
target = np.array(target)

# Plot the images side by side
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(target)
ax[1].set_title('Segmentation Mask')
ax[2].imshow(segmentation)
ax[2].set_title('Predicted Segmentation')

plt.show()




#сделать какую то оценку обучения модели
#реализовать другие методы
