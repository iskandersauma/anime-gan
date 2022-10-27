import os
import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from model import Generator

def load_image(path, size=None):
    image = image2tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    w, h = image.shape[-2:]
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size)//2
        right = left + crop_size
        top = (h - crop_size)//2
        bottom = top + crop_size
        image = image[:,:,left:right, top:bottom]

    if size is not None and image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)
    
    return image

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()
    
device = 'cpu'
torch.set_grad_enabled(False)
image_size = 800 # Can be tuned, works best when the face width is between 200~250 px

model = Generator().eval().to(device)

ckpt = torch.load(f"weights/celeba_distill.pt", map_location=device)
model.load_state_dict(ckpt)

image = load_image(f"/home/iskander/Documents/animegan2-pytorch/IMG-20220218-WA0008.jpg", image_size)
output = model(image.to(device))
results = []
results.append(torch.cat([image, output.cpu()], 3))
results = torch.cat(results, 2)

imshow(tensor2image(results),40)
cv2.imwrite('face_results.jpg', cv2.cvtColor(255*tensor2image(results), cv2.COLOR_BGR2RGB))
