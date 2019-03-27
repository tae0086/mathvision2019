import numpy as np


def rigid_transform(p: np.ndarray, p1: np.ndarray, p1_prime: np.ndarray, r1: np.ndarray, r2: np.ndarray):
    return np.transpose(np.matmul(np.matmul(r1, np.transpose(p - p1)), r2)) + p1_prime


def rotational_transform(axis: np.ndarray, cos: float):
    sin = np.sqrt(1 - cos ** 2)
    transform = np.asarray([
        [cos + axis[0] ** 2 * (1 - cos), axis[0] * axis[1] * (1 - cos) - axis[2] * sin,
         axis[0] * axis[2] * (1 - cos) + axis[1] * sin],
        [axis[1] * axis[0] * (1 - cos) + axis[2] * sin, cos + axis[1] ** 2 * (1 - cos),
         axis[1] * axis[2] * (1 - cos) - axis[0] * sin],
        [axis[2] * axis[0] * (1 - cos) - axis[1] * sin, axis[2] * axis[1] * (1 - cos) + axis[0] * sin,
         cos + axis[2] ** 2 * (1 - cos)]
    ])
    return transform


def cos(vec1: np.ndarray, vec2: np.ndarray):
    return np.inner(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def unit_vector(vec: np.ndarray):
    return vec / np.linalg.norm(vec)


# points in A
p1 = np.asarray([-0.5, 0.0, 2.121320])
p2 = np.asarray([0.5, 0.0, 2.121320])
p3 = np.asarray([0.5, -0.707107, 2.828427])

# points in A prime
p1_prime = np.asarray([1.363005, -0.427130, 2.339082])
p2_prime = np.asarray([1.748084, 0.437983, 2.017688])
p3_prime = np.asarray([2.636461, 0.184843, 2.400710])

# vectors in A
p1_p2 = p2 - p1
p1_p3 = p3 - p1

# vectors in A prime
p1_prime_p2_prime = p2_prime - p1_prime
p1_prime_p3_prime = p3_prime - p1_prime

# normal vector in A and A prime
h = np.cross(p1_p2, p1_p3)
h_prime = np.cross(p1_prime_p2_prime, p1_prime_p3_prime)

# rotational transform: R1
u = unit_vector(np.cross(h, h_prime))  # axis of rotation in R1
cos_theta = cos(h, h_prime)
r1 = rotational_transform(axis=u, cos=cos_theta)

# R1 validation
# val_out = np.matmul(r1, h)
# print('R1 calculation: {}'.format(val_out))
# print('R1 ground truth: {}\n'.format(h_prime))

# intermediate result
inter_out = np.matmul(r1, p1_p3)

# rotational transform: R2
u = unit_vector(h_prime)  # axis of rotation in R2
cos_theta = cos(inter_out, p1_prime_p3_prime)
r2 = rotational_transform(axis=u, cos=cos_theta)

# R2 validation
val_out = np.matmul(inter_out, r2)
print('R2 calculation: {}'.format(val_out))
print('R2 ground truth: {}\n'.format(p1_prime_p3_prime))

# p1, p2, p3 result
p1_out = rigid_transform(p1, p1, p1_prime, r1, r2)
print('p1 calculation: {}'.format(p1_out))
print('p1 ground truth: {}\n'.format(p1_prime))

p2_out = rigid_transform(p2, p1, p1_prime, r1, r2)
print('p2 calculation: {}'.format(p2_out))
print('p2 ground truth: {}\n'.format(p2_prime))

p3_out = rigid_transform(p3, p1, p1_prime, r1, r2)
print('p3 calculation: {}'.format(p3_out))
print('p3 ground truth: {}\n'.format(p3_prime))

# p4 result
p4 = np.asarray([0.5, 0.707107, 2.828427])
p4_prime = np.asarray([1.4981, 0.8710, 2.8837])
p4_out = rigid_transform(p4, p1, p1_prime, r1, r2)
print('p4 calculation: {}'.format(p4_out))
print('p4 ground truth: {} \n'.format(p4_prime))

# p5 result
p5 = np.asarray([1, 1, 1])
p5_out = rigid_transform(p5, p1, p1_prime, r1, r2)
print('p5 calculation: {}'.format(p5_out))
