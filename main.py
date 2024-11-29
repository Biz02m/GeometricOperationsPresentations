import requests
import numpy as np
import cv2
from transformations import *

SAVE_IMAGE = True
SHOW_IMAGE = False

path_to_image = "./images/"
cv2_path = "cv2_processed/"
my_implementation_path = "my_implementation/"

image_url = "https://m.media-amazon.com/images/I/61tBzoz6c0L._AC_SX679_.jpg"
response = requests.get(image_url)

if response.status_code != 200:
    print("pobieranie obrazu nie powiodło się")
    exit(1)

image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

height, width = image.shape[:2]
output_height, output_width = (4 * height) // 3, (4 * width) // 3

# parametry translacji
tx = 100
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])

# parametry rotacji
center_x, center_y = width // 2, height // 2
angle = 45
scale = 0.5
R = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)

# parametry skalowania
new_width, new_height = width // 2, height // 2
# new_width, new_height = width * 2, height * 2


# parametr odbicia
flipCode = 0

# parametry ścinania
shear = 0.5
S = np.float32([[1, shear, 0], [0, 1, 0]])

# parametry transformacji affinicznej
src_points = np.float32([[50, 50], [200, 50], [50, 200]])
dst_points = np.float32([[10, 100], [220, 50], [100, 250]])
A = cv2.getAffineTransform(src_points, dst_points)

# parametry transformacji perspektywicznej
src_points_perspective = np.float32([[50, 50], [width - 50, 50], [50, height - 50], [width - 50, height - 50]])
dst_points_perspective = np.float32([[100, 100], [width - 100, 50], [100, height - 100], [width - 100, height - 50]])
P = cv2.getPerspectiveTransform(src_points_perspective, dst_points_perspective)

# transformacje z użyciem biblioteki opencv
translated_cv2 = cv2.warpAffine(image, M, (width, height))
resized_cv2 = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
rotated_cv2 = cv2.warpAffine(image, R, (width, height))
flipped_cv2 = cv2.flip(image, flipCode)
sheared_cv2 = cv2.warpAffine(image, S, (output_width, output_height))
affine_transformed_cv2 = cv2.warpAffine(image, A, (output_width, output_height))
perspective_transformed_cv2 = cv2.warpPerspective(image, P, (width, height))

# transformacje z własną implementacją
translated = translate_image(image, tx, ty)
rotated = rotate_image(image, angle)
scaled = resize_image(image, 2)
sheared = shear_image(image, shear)
flipped = flip_image(image)

if SAVE_IMAGE:
    cv2.imwrite(path_to_image + cv2_path + "original_cv2.jpg", image)
    cv2.imwrite(path_to_image + cv2_path + "translated_cv2.jpg", translated_cv2)
    cv2.imwrite(path_to_image + cv2_path + "resized_cv2.jpg", resized_cv2)
    cv2.imwrite(path_to_image + cv2_path + "rotated_cv2.jpg", rotated_cv2)
    cv2.imwrite(path_to_image + cv2_path + "flipped_cv2.jpg", flipped_cv2)
    cv2.imwrite(path_to_image + cv2_path + "sheared_cv2.jpg", sheared_cv2)
    cv2.imwrite(path_to_image + cv2_path + "affine_cv2.jpg", affine_transformed_cv2)
    cv2.imwrite(path_to_image + cv2_path + "perspective_cv2.jpg", perspective_transformed_cv2)

    # my implementation
    cv2.imwrite(path_to_image + my_implementation_path + "translated.jpg", translated)
    cv2.imwrite(path_to_image + my_implementation_path + "rotated.jpg", rotated)
    cv2.imwrite(path_to_image + my_implementation_path + "scaled.jpg", scaled)
    cv2.imwrite(path_to_image + my_implementation_path + "sheared.jpg", sheared)
    cv2.imwrite(path_to_image + my_implementation_path + "flipped.jpg", flipped)

if SHOW_IMAGE:
    cv2.imshow("Oryginalny Obraz", image)
    cv2.imshow("Przesunięty Obraz cv2", translated_cv2)
    cv2.imshow("Zmniejszony Obraz cv2", resized_cv2)
    cv2.imshow("Obrócony Obraz cv2", rotated_cv2)
    cv2.imshow("Odbity Obraz cv2", flipped_cv2)
    cv2.imshow("Ścięty Obraz cv2", sheared_cv2)
    cv2.imshow("Transformacja affiniczna cv2", affine_transformed_cv2)
    cv2.imshow("Transformacja perspektywiczna cv2", perspective_transformed_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
