import sys
sys.path.insert(0, '../')
from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def parseArrayDataToInt(arr = []):
    newArr = []
    for i in arr:
        line = []
        for j in i:
            data = []
            for d in j:
                data.append(int(d))
            line.append(data)
        newArr.append(line)
    return newArr


#%matplotlib inline

# read the image
img = plt.imread('casa.jpg')

# reshaping the pixels matrix
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))

# SOM initialization and training
print('training...')
som = MiniSom(3, 3, 3, sigma=1.,
              learning_rate=0.2, neighborhood_function='bubble')  # 3x3 = 9 final colors
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()  # saving the starting weights
som.train_random(pixels, 100,True)

print('quantization...')
qnt = som.quantization(pixels)  # quantize each pixels of the image
print('building new image...')

clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):  # place the quantized values into a new image
    clustered[np.unravel_index(i, shape=(img.shape[0], img.shape[1]))] = q/255
print('done.')


# show the result
plt.figure(figsize=(7, 7))
plt.figure(1)
plt.subplot(221)
plt.title('original')
plt.imshow(img)
plt.subplot(222)
plt.title('result')
plt.imshow(clustered)


starting_weights  = parseArrayDataToInt(starting_weights)
final_weights = parseArrayDataToInt(som.get_weights())

plt.subplot(223)
plt.title('initial colors')
plt.imshow(starting_weights, interpolation='none')
plt.subplot(224)
plt.title('learned colors')
plt.imshow(final_weights, interpolation='none')

plt.tight_layout()
plt.show()