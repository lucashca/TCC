import sys
from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

maxEpochs = 100
somSize = 10
try:
    arg = sys.argv[1:]
    
    if(arg[0]):
        maxEpochs = int(arg[0])
    if(arg[1]):
        somSize = int(arg[1])

except :
    pass





data = np.genfromtxt('./datasets/solo.csv', delimiter=',')
# Elimine header
data = data[1:]

print(len(data[0]))


data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 0, data)

'''
# Initialization and training
som = MiniSom(somSize, somSize, len(data[0]), sigma=2.5, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=123)

som.pca_weights_init(data)
print("Training...")
som.train_batch(data, maxEpochs, verbose=True)  # random training
print("\n...ready!")



plt.figure(figsize=(7, 7))
# Plotting the response for each pattern in the iris dataset

plt.subplot(2,1,1)
plt.pcolor(som.distance_map().T, cmap='bone')  # plotting the distance map as background
plt.colorbar()

plt.subplot(2,1,2)
frequencies = np.zeros((somSize, somSize))
for position, values in som.win_map(data).items():
    frequencies[position[0], position[1]] = len(values)
plt.pcolor(frequencies, cmap='Purples')
plt.colorbar()
plt.show()

'''


som = MiniSom(somSize, somSize, len(data[0]), sigma=1., learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=10)
som.pca_weights_init(data)
max_iter = maxEpochs


q_error_pca_init = []
iter_x = []

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.ylabel('quantization error')
plt.xlabel('iteration index')

def animate(i):
    
    global som,max_iter,q_error_pca_init,iter_x,ax1
    percent = 100*(i+1)/max_iter
    rand_i = np.random.randint(len(data))
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    
    error = som.quantization_error(data)
    q_error_pca_init.append(error)
    iter_x.append(i)
    ax1.clear()
    ax1.pcolor(som.distance_map().T, cmap='bone')
    sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')



ani = animation.FuncAnimation(fig,animate, interval=1) 
plt.show()


