from albumentations import (
    Compose,
    RandomRotate90,
    Flip,
    Transpose,
    RandomSizedCrop,
    RandomBrightnessContrast,
    RandomGamma,
    Normalize
)

def augment(image):
    augmentations = Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        RandomBrightnessContrast(p = 0.5),
        RandomGamma(p = 0.5),
        Normalize()
    ])

    augmented = augmentations(image = image)
    image = augmented['image']

    return image