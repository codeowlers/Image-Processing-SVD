import time
import cv2
def save_image(data):
    current_timestamp = int(time.time())

    # Save the compressed image
    cv2.imwrite(f'compressed-images/{current_timestamp}.jpg', data)
