from scipy.io import loadmat 
import matplotlib.pyplot as plt
import numpy as np
from lr_train import lr_train

# load data
mat = loadmat('breast-cancer-data.mat')

# store data
x = np.matrix(mat['X'])
y = np.matrix(mat['Y'], dtype=np.dtype('i4')) # to change Y = 0 to Y = -1

step_size_range = [1, 0.1, 0.001, 0.0001, 1e-5]
c = 10e-3
xb = np.hstack((np.ones((len(y), 1)), x))
x_axis = np.arange(0, 5000, 1)

gradnorm_arr = [len(step_size_range)]
# w, obj, gradnorm = lr_train(x, y, c, step_size_range[0], 0, 5000)

for i in range(len(step_size_range)):
    w, obj, gradnorm = lr_train(x, y, c, step_size_range[i], 0, 5000)
    #weights[i] = w
    plt.plot(x_axis, obj)
    plt.yscale('symlog')
    
plt.show()
# # x axis values 
# x = [1,2,3] 
# # corresponding y axis values 
# y = [2,4,1] 
  
# # plotting the points  
# plt.plot(x, y) 
  
# # naming the x axis 
# plt.xlabel('x - axis') 
# # naming the y axis 
# plt.ylabel('y - axis') 
  
# # giving a title to my graph 
# plt.title('My first graph!') 
  
# # function to show the plot 
# plt.show()