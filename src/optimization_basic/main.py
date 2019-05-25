import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
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
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
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
# print('({}, {})'.format(z_gradient[0].subs([(x, 0), (y, 0)]), z_gradient[1].subs([(x, 0), (y, 0)])))

# Q1-3
# find critical points
critical_points = solve_poly_system(z_gradient, [x, y])
print(critical_points)  # [(0, -1), (0, 0), (3/8, -3/4), (1, -1)]

# hessian matrix
hessian = [[z.diff(x).diff(x), z.diff(x).diff(y)], [z.diff(y).diff(x), z.diff(y).diff(y)]]
# print(hessian)

# second partial derivative test
for critical_x, critical_y in critical_points:
    f_xx = hessian[0][0].subs([(x, critical_x), (y, critical_y)])
    f_xy = hessian[0][1].subs([(x, critical_x), (y, critical_y)])
    f_yx = hessian[1][0].subs([(x, critical_x), (y, critical_y)])
    f_yy = hessian[1][1].subs([(x, critical_x), (y, critical_y)])
    assert f_xy == f_yx, 'f_xy and f_yx must be same!'

    # determinant of hessian
    determinant = f_xx * f_yy - f_xy ** 2

    result = ''

    if determinant > 0 and f_xx > 0:  # local minimum
        result = 'local minimum'
    elif determinant > 0 > f_xx:  # local maximum
        result = 'local maximum'
    elif determinant < 0:  # saddle point
        result = 'saddle point'
    elif determinant == 0:
        result = 'inconclusive'

    print('* critical point ({}, {}) is {}'.format(critical_x, critical_y, result))

# plot function and critical points simultaneously
styles = ['co', 'mo', 'yo', 'ko']
for i in range(len(styles)):
    ax.plot([critical_points[i][0]], [critical_points[i][1]], styles[i])
ax.view_init(0, 0)  # rotation
plt.show()
