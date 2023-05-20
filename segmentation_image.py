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
import numpy as np

#Определение модели CRF-RNN
class CRF_RNN(nn.Module):
    def __init__(self, num_classes):
        super(CRF_RNN, self).__init__()
        self.num_classes = num_classes

        # Определение CNN
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

        # Определение CRF-RNN
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

#Преобразование изображений в тензоры
transform = transforms.Compose([
    transforms.ToTensor()  # convert image to tensor
])

# Функция для объединения элементов батча
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

#загрузка датасетов
train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=False, transform=transform)
test_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=transform)


# Определение загрузчиков данных
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

# Определение модели
model = CRF_RNN(num_classes=21)
#Загрузка весов
checkpoint = torch.load('model_55EPOH.pt')
model.load_state_dict(checkpoint)


# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#Обучение модели
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
##Сохранение весов обученной модели
# torch.save(model.state_dict(), 'model_55EPOH.pt')
# Define the device to run the model on


# Перенос модели на устройство (GPU, если доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#Вывод результатов
index = 155

#Выбираем изображение и маску
image, target = test_dataset[index]

#  Создание мини-батча с одним изображением
images = image.unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = images.to(device)

#Результат модели
with torch.no_grad():
    outputs = model(images)

#Преобразование предсказаний в карту сегментации
_, predicted = torch.max(outputs.data, 1)
segmentation = predicted.cpu().numpy()[0]

#Получение  исходного изображения
image = images.cpu().numpy()[0].transpose(1, 2, 0)

# маска
target = np.array(target)

#Вывод изображений на экран
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(image)
ax[0].set_title('Оригинальное изображение')
ax[1].imshow(target)
ax[1].set_title('Сегментированное изображение')
# ax[2].imshow(segmentation)
# ax[2].set_title('Predicted Segmentation')

plt.show()

