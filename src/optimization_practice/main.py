import random

import numpy as np
from colour import Color
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sympy import symbols, sin, solve

# 1. Draw a graph in 3d space
x = np.linspace(start=-1, stop=5, num=30)
y = np.linspace(start=-3, stop=4, num=30)
x, y = np.meshgrid(x, y)
z = np.sin(x + y - 1) + (x - y - 1) ** 2 - 1.5 * x + 2.5 * y + 1

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z, cmap='gist_earth', antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# ax.view_init(10, 40)  # rotation
# plt.show()

# 2. Optimization: gradient descent method
x, y = symbols('x y')
f = sin(x + y - 1) + (x - y - 1) ** 2 - 1.5 * x + 2.5 * y + 1
f_gradient = [f.diff(x), f.diff(y)]
# print(f_gradient)

history = [(random.uniform(-1, 5), random.uniform(-3, 4))]  # random initialization

iteration = 20
lambda_ = 0.1

for i in range(iteration):
    gradient = (f_gradient[0].subs([(x, history[i][0]), (y, history[i][1])]),
                f_gradient[1].subs([(x, history[i][0]), (y, history[i][1])]))
    step = tuple(lambda_ * derivative for derivative in gradient)
    history.append((history[i][0] - step[0], history[i][1] - step[1]))

# Plot gradient descent history
colors = list(Color('red').range_to(Color('green'), iteration + 1))
for i, point in enumerate(history):
    ax.plot([point[0]], [point[1]], [f.subs([(x, point[0]), (y, point[1])])], marker='o', color=colors[i].get_hex_l())

# Show the first and last point
ax.text(x=5, y=-4, z=65, s='initial point:    '
                           '({:.1f}, {:.1f}, {:.1f})'.format(history[0][0], history[0][1],
                                                             f.subs([(x, history[0][0]),
                                                                     (y, history[0][1])])), fontsize=12)
ax.text(x=5, y=-4, z=60, s='optimal point: '
                           '({:.1f}, {:.1f}, {:.1f})'.format(history[-1][0], history[-1][1],
                                                             f.subs([(x, history[-1][0]),
                                                                     (y, history[-1][1])])), fontsize=12)

# ax.view_init(30, 40)  # rotation
# plt.show()

# 3. Optimization: newton's method
hessian = [
    [f.diff(x).diff(x), f.diff(x).diff(y)],
    [f.diff(y).diff(x), f.diff(y).diff(y)]
]
# print(hessian)
