import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

# Загрузка изображения
test_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=False)
index = 116
transform = transforms.ToTensor()
image, target = test_dataset[index]
image = transform(image)
image = np.transpose(image.numpy(), (1, 2, 0))  # Преобразование тензора в массив NumPy

# Приведение изображения к одномерному массиву пикселей
pixels = image.reshape(-1, 3)

# Кластеризация пикселей методом k-средних
kmeans = KMeans(n_clusters=2)  # Здесь можно задать желаемое количество кластеров
kmeans.fit(pixels)

# Получение меток кластеров для каждого пикселя
labels = kmeans.labels_

# Преобразование меток в двумерный массив, соответствующий размеру исходного изображения
segmented_image = labels.reshape(image.shape[:2])

# Визуализация результата
plt.imshow(segmented_image)
plt.axis('off')
plt.show()


fig, ax = plt.subplots(ncols=2)
ax[0].imshow(image)
ax[0].set_title('Исходное изображение')
ax[1].imshow(segmented_image)
ax[1].set_title('Сегментированное изображение')

plt.show()