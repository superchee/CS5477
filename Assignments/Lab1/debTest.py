
#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np


from lab1 import *


#%%
np.set_printoptions(precision=6)  # Print less digits


input_pts = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5],
                      [1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 0.5]])

cases = ['translation', 'rigid', 'affine', 'homography']
h = {}
h['translation'] = np.array([[1.0, 0.0, 1.0],
                             [0.0, 1.0, 0.5],
                             [0.0, 0.0, 1.0]])
h['rigid'] = np.array([[0.8660254, -0.5, 0.5],
                       [0.5, 0.8660254, 0.1],
                       [0.0, 0.0, 1.0]])
h['affine'] = np.array([[1.0, 0.5, 0.0],
                        [0.0, 1.5, 0.0],
                        [0.0, 0.0, 1.0]])
h['homography'] = np.array([[1.0, 0.2, 0.0],
                       [0.0, 1.5, 0.0],
                       [0.0, 0.5, 1.0]])

output_pts = {}
for c in cases:
    output_pts[c] = transform_homography(input_pts, h[c])

#%%
# Print output points and plot
for c in cases:
    print('Points after ({})\n'.format(c), output_pts[c].transpose())
    plt.figure(figsize=(12,5))
    ax = plt.subplot(1, 2, 1)
    ax.axis('equal')
    ax.set(xlim=(-1.0, 3.0), ylim=(-1.0, 3.0))
    plt.title('Before {}'.format(c))
    plt.plot(input_pts[:, 0], input_pts[:, 1], '*')
    ax = plt.subplot(1, 2, 2)
    ax.axis('equal')
    ax.set(xlim=(-1.0, 3.0), ylim=(-1.0, 3.0))

    plt.title('After {}'.format(c))
    plt.plot(output_pts[c][:, 0], output_pts[c][:, 1], '*');




#%%
