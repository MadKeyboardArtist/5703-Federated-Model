from torchvision import transforms
from PIL import Image


def resize_with_padding(img, image_size=224):
    # 1. get original size
    w, h = img.size
    # 2. scale so the *longer* side becomes image_size
    scale = image_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # 3. create a black square canvas
    new_img = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    
    # 4. paste resized image centered
    left = (image_size - new_w) // 2
    top  = (image_size - new_h) // 2
    new_img.paste(img, (left, top))

    return new_img