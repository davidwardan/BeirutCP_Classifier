import numpy as np
import matplotlib.pyplot as plt

# Load the images from a .npy file
images = np.load('/Users/davidwardan/PycharmProjects/BeirutCP_Classifier/input/test/x_test.npy'
                 , allow_pickle=True)  # Load the array of images

# Get shape of the images
print(images.shape)

# Dsiplay first n images in the dataset in a grid format
n = 100
img_per_row = 20
plt.figure(figsize=(20, 5))
for i in range(n):
    plt.subplot(int(n / img_per_row), img_per_row, i + 1)
    plt.imshow(images[i])
    plt.axis('off')

plt.savefig('100_images.png', bbox_inches='tight')
