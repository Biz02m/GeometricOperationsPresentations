import numpy as np


def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Centrum obrazu
    angle_rad = np.deg2rad(angle)

    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    rotated_image = np.zeros_like(image)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # Przekształcenie współrzędnych
            new_coords = np.dot(rotation_matrix, np.array([x - center[0], y - center[1]])) + center
            new_x, new_y = new_coords[0].astype(int), new_coords[1].astype(int)

            # Sprawdzamy, czy nowe współrzędne są w obrębie obrazu
            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                rotated_image[new_y, new_x] = image[y, x]

    return rotated_image


def translate_image(image, tx, ty):
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    translated_image = np.zeros_like(image)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            coords = np.array([x, y, 1])
            new_coords = np.dot(translation_matrix, coords)
            new_x = new_coords[0].astype(int)
            new_y = new_coords[1].astype(int)

            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                translated_image[new_y, new_x] = image[y, x]

    return translated_image


def resize_image(image, scale):
    scaling_matrix = np.array([[scale, 0],
                               [0, scale]])

    scaled_image = np.zeros_like(image)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            coords = np.array([x, y])
            new_coords = np.dot(scaling_matrix, coords)
            new_x = new_coords[0].astype(int)
            new_y = new_coords[1].astype(int)

            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                scaled_image[new_y, new_x] = image[y, x]

    return scaled_image


def shear_image(image, mx):
    shear_matrix = np.array([[1, mx],
                             [0, 1]])

    sheared_image = np.zeros_like(image)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            coords = np.array([x, y])
            new_coords = np.dot(shear_matrix, coords)
            new_x = new_coords[0].astype(int)
            new_y = new_coords[1].astype(int)

            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                sheared_image[new_y, new_x] = image[y, x]

    return sheared_image


def flip_image(image):
    flip_matrix = np.array([[-1, 0],
                            [0, 1]])

    offset = np.array([image.shape[1] - 1, 0])

    flipped_image = np.zeros_like(image)

    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            coords = np.array([x, y])
            new_coords = np.dot(flip_matrix, coords) + offset
            new_x = new_coords[0].astype(int)
            new_y = new_coords[1].astype(int)

            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
                flipped_image[new_y, new_x] = image[y, x]

    return flipped_image
