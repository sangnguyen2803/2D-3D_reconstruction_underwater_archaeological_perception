import cv2


def resize_rgb(image, max_size=960, enhance=False):
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(w, h))
    width, height = round(w * scale), round(h * scale)
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if enhance:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # Rounded sizes require different exact scale factors on each axis.
    return result, (width / w, height / h)
