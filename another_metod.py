from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np


test_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=False)

# Определите индекс, соответствующий нужному изображению
index = 131

# Преобразование изображений в тензоры
transform = transforms.ToTensor()

# Получите нужное изображение по индексу и примените преобразование в тензор
image, target = test_dataset[index]
image = transform(image)

# Преобразуйте изображение в оттенки серого
gray_image = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY).astype(np.uint8)

# Определите пороговое значение с помощью метода Оцу
_, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Вывод результата
plt.subplot(1, 2, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title('Исходное изображение')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Сегментированное изображение')

plt.tight_layout()
plt.show()

