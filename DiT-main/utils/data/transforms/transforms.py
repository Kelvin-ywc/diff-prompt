from torchvision import transforms as T

def build_trans(is_train=True, image_size=256):
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    if is_train:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

def build_mask_trans(is_train=True, image_size=256):
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
