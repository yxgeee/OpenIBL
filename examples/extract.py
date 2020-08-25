import torch
from torchvision import transforms
from PIL import Image

# load the best model with PCA (trained by our SFRS)
model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()

# read image
img = Image.open('image.jpg').convert('RGB') # modify the image path according to your need
transformer = transforms.Compose([transforms.Resize(480, 640), # (height, width)
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
img = transformer(img)

# use GPU (optional)
mdoel = model.cuda()
img = img.cuda()

# extract descriptor (4096-dim)
with torch.no_grad():
    des = model(img.unsqueeze(0))[0]
des = des.cpu().numpy()
