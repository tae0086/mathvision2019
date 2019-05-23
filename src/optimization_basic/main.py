import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sympy import symbols, solve_poly_system


def f(x: np.ndarray, y: np.ndarray):
    """Scalar fucntion for x and y."""
    return (x + y) * (x * y + x * y ** 2)


# Q1-1
# define x, y, z as numpy array
x = np.linspace(start=-2, stop=2, num=30)
y = np.linspace(start=-2, stop=2, num=30)
x, y = np.meshgrid(x, y)  # coordinate matrix from x and y vectors
z = f(x, y)

# plot 3d surface
figure = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', antialiased=True)

# label
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plt.show()

# Q1-2
# define function using symbol x and y
x, y = symbols('x y')
z = (x + y) * (x * y + x * y ** 2)
z_gradient = [z.diff(x), z.diff(y)]  # gradient of z
# print(z_gradient)
# TODO: 검토 필요 - (0, 0)에서의 gradient가 (0, 0)
print('({}, {})'.format(z_gradient[0].subs([(x, 0), (y, 0)]), z_gradient[1].subs([(x, 0), (y, 0)])))

# Q1-3
# find critical points
print(solve_poly_system(z_gradient, [x, y]))  # [(0, -1), (0, 0), (3/8, -3/4), (1, -1)]
