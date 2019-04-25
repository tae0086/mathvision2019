from os import listdir
from os.path import isdir

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from sklearn.decomposition import PCA


def read_gray_image(path: str):
    """Read gray image and flatten(1-D)."""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.flatten()
    return image


if __name__ == '__main__':
    # configuration
    num_of_classes = 40
    root_dir = './data/att_faces/'
    train_dir = root_dir + 'train/'
    test_dir = root_dir + 'test/'

    # all train faces ordered by class
    faces = []

    # read face images split by class directory
    for i in range(num_of_classes):
        class_dir = train_dir + '{}/'.format(i + 1)
        for filename in listdir(class_dir):
            filepath = class_dir + filename
            if isdir(filepath):
                continue
            face = read_gray_image(filepath)
            faces.append(face)

    # normalize (convert to ndarray and normalize)
    faces = np.array(faces)
    faces -= faces.mean(axis=0).round().astype('uint8')

    # PCA for k=1, 10, 100, 200
    num_of_eigenfaces = [1, 10, 100, 200]
    pca = {}

    for k in num_of_eigenfaces:
        pca[k] = PCA(n_components=k)
        pca[k].fit(faces)

    # eigenfaces visualization for k=10
    eigenfaces = pca[10].components_
    plt.figure(1)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(eigenfaces[i].reshape(56, 46))
        plt.axis('off')

    plt.show()

    # in 22th person, reconstruction using first image
    test_face = read_gray_image(test_dir + 's22_1.png')
    reconstructions = {}

    for k in pca:
        test_face_low = np.matmul(test_face, pca[k].components_.T)
        # check if lengths of two(weights and eigenfaces) are same
        if len(test_face_low) != len(pca[k].components_):
            print('length of weight vector and eigenface matrix are not same.')
            break

        reconstruction = sum([test_face_low[i] * pca[k].components_[i] for i in range(len(pca[k].components_))])
        reconstructions[k] = reconstruction

    # visualize reconstruction faces
    plt.figure(2)

    for i in range(4):
        k = num_of_eigenfaces[i]
        reconstruction = reconstructions[k]

        plt.subplot(1, 4, i + 1)
        plt.imshow(reconstruction.reshape(56, 46))
        plt.title('k={}'.format(k))
        plt.axis('off')

    plt.show()

    # all test faces ordered by class
    test_faces = []

    for i in range(num_of_classes):
        filepath = test_dir + 's{}_1.png'.format(i + 1)
        test_face = read_gray_image(filepath)
        test_faces.append(test_face)

    # normalize test faces
    test_faces = np.array(test_faces)
    test_faces -= test_faces.mean(axis=0).round().astype('uint8')

    # classify test faces projected on subspace that has k=1, 10, 100, 200 dimensions
    for k in pca:
        # projection on subspace
        faces_low = np.matmul(faces, pca[k].components_.T)
        test_faces_low = np.matmul(test_faces, pca[k].components_.T)

        # classification results for all test faces (class: 0~39)
        classifications = []

        # average of distance
        for test_face_low in test_faces_low:
            # distance per class
            distances = []

            # class loop
            for i in range(num_of_classes):
                # euclidean distance
                euclidean = np.array(
                    [np.linalg.norm(test_face_low - face_low) for face_low in faces_low[9 * i:9 * (i + 1), :]]
                ).mean()
                # cosine distance
                # cosine = np.array(
                #     [spatial.distance.cosine(test_face_low, face_low) for face_low in faces_low[9 * i:9 * (i + 1), :]]
                # ).mean()
                distances.append(euclidean)
                # distances.append(cosine)

            classifications.append(np.argmin(distances))

        # accuracy
        corrects = sum([1 if i == c else 0 for i, c in enumerate(classifications)])
        accuracy = corrects / len(classifications)
        print('in k={}, accuracy is {}.'.format(k, accuracy))

    # bar graph for classification results
    # configuration
    x = [1, 10, 100, 200]
    y = [17.5, 67.5, 77.5, 77.5]
    offset_x = -0.1
    offset_y = 0.5

    plt.figure(3)
    plt.bar(range(4), y, width=0.5)
    # put value text on bar
    for a, b in zip(range(4), y):
        plt.text(a + offset_x, b + offset_y, str(b))
    plt.xticks(range(4), x)
    plt.xlabel('k')
    plt.ylabel('accuracy (%)')
    plt.show()

    # additional experiment 1: k=50, 60, 70, 80, 90
    # configuration
    x = [50, 60, 70, 80, 90, 100]
    y = [75, 75, 75, 77.5, 77.5, 77.5]
    offset_x = -0.2

    plt.figure(4)
    plt.bar(range(6), y, width=0.5)
    # put value text on bar
    for a, b in zip(range(6), y):
        plt.text(a + offset_x, b + offset_y, str(b))
    plt.xticks(range(6), x)
    plt.xlabel('k')
    plt.ylim(70, 80)
    plt.ylabel('accuracy (%)')
    plt.show()

    # additional experiment 2: cosine distance VS euclidean distance
    # configuration
    x = [1, 10, 100, 200]
    y_cosine = [0, 60, 55, 47.5]
    y_euclidean = [17.5, 67.5, 77.5, 77.5]

    plt.figure(5)
    plt.plot(range(4), y_cosine, 'r-', label='cosine')
    plt.plot(range(4), y_euclidean, 'b-', label='euclidean')
    plt.xticks(range(4), x)
    plt.xlabel('k')
    plt.ylabel('accuracy (%)')
    plt.legend()
    plt.show()
