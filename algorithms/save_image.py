import time
import cv2
import os

def save_image(data, image_type, onlySize=False):
    current_timestamp = int(time.time())
    path = f'compressed-images/{current_timestamp}.{image_type}'
    # Save the compressed image
    # Save the compressed image
    if image_type == 'jpg':
        cv2.imwrite(path, data, [cv2.IMWRITE_JPEG_QUALITY, 80])
    elif image_type == 'png':
        cv2.imwrite(path, data, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif image_type == 'webp':
        cv2.imwrite(path, data, [cv2.IMWRITE_WEBP_QUALITY, 80])
    else:
        raise ValueError(f'Invalid image format: {format}')

    size = os.path.getsize(path)
    # Convert size to MB
    size = round(size / (1024.0 * 1024.0), 4)
    if onlySize is True:
        # Delete the saved image
        os.remove(path)   
    else:
        print("Image saved in path: " ,path)

    print(f"Image size: {size} mb")


    return size, path
