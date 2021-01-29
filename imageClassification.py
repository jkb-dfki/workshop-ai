# System Imports
import sys

# Third-Party Imports
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("classes") as cls_file:
    classes = cls_file.read().split("\n")

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

model = torchvision.models.alexnet(pretrained=True)
model.eval().to(device)

image = Image.open(sys.argv[1])
image = transform(image).to(device)
image = image.unsqueeze(0)

outputs = model(image)

print(classes[torch.argmax(outputs).detach().cpu().numpy()])
