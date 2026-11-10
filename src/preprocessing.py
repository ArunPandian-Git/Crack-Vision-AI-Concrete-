import cv2

image = cv2.imread("/3916.rf.842ca41d6f6e2c0f89b62d6245f5a6ae.jpg")

resized_image = cv2.resize(
    image,
    (640, 640)
)

print(resized_image.shape)
