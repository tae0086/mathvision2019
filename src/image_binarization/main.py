import cv2
import numpy as np
from matplotlib import pyplot as plt


def surface_approximation_at(row_index: int, column_index: int, solution: np.ndarray):
    """Calculate quadratic surface approximation value at specific pixel of image."""
    a, b, c, d, e, f = solution
    x, y = row_index + 1, column_index + 1
    return a * x * x + b * y * y + c * x * y + d * x + e * y + f


def image_surface_approximation(image: np.ndarray, solution: np.ndarray):
    """Approximate image background to quadratic surface."""
    background = np.empty(shape=image.shape)
    for indices, value in np.ndenumerate(image):
        row_index, column_index = indices
        background[row_index][column_index] = surface_approximation_at(row_index, column_index, solution)

    # visualize approximation result
    # plt.imshow(background, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return background


def normalization(vec: np.ndarray):
    """Normalize vector value between 0 and 1."""
    return (vec - vec.min()) / (vec.max() - vec.min())


def image_binarization(image: np.ndarray):
    threshold = -10
    binarized = np.empty(shape=image.shape)

    for indices, value in np.ndenumerate(image):
        row_index, column_index = indices
        binarized[row_index][column_index] = 0 if image[row_index][column_index] > threshold else 255

    # visualize binarized image
    plt.imshow(binarized, cmap='gray')
    plt.title('threshold={}'.format(threshold))
    plt.axis('off')
    plt.show()

    return binarized


# read image and flatten
image = cv2.imread('hw10_sample.png', cv2.IMREAD_GRAYSCALE)
image_flatten = image.flatten()
height, width = image.shape[:2]

# make A matrix (Ax = b)
A = np.empty(shape=(width * height, 6))

for indices, _ in np.ndenumerate(image):
    row_index, column_index = indices
    x, y = row_index + 1, column_index + 1
    A[width * row_index + column_index] = [x * x, y * y, x * y, x, y, 1]

# pseudo inverse of A
A_pinv = np.linalg.pinv(A)

# evaluate x vector (x = A^+b)
x_vec = np.matmul(A_pinv, image_flatten)
# print(x_vec)
background = image_surface_approximation(image, x_vec)

# brightness correction
correction_image = image - background

# visualize correction image
# plt.imshow(correction_image, cmap='gray')
# plt.axis('off')
# plt.show()

# intensity histogram of correction image
# plt.hist(x=correction_image.flatten(), bins=256, density=True)
# plt.show()

# image binarization
image_binarization(correction_image)
