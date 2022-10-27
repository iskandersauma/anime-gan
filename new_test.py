import torch

from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image
cuda = torch.device('cuda')     


img = Image.open("Iskander_Sauma_auf219.jpg")
convert_tensor = transforms.ToTensor()
img = convert_tensor(img).cuda()
img = torch.unsqueeze(img,0)


pretrained_options = ["celeba_distill", "face_paint_512_v1", "face_paint_512_v2", "paprika"]

for pretrained_model in pretrained_options:
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained=pretrained_model, device="cuda")
    out = model(img)

    save_image(out, f'photo_{pretrained_model}.png')
