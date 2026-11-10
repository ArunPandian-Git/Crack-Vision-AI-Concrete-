import cv2

image = cv2.imread("sample.jpg")

resized_image = cv2.resize(
    image,
    (640, 640)
)

print(resized_image.shape)
