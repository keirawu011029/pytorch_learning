from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("./logs")
img = Image.open("data/train/ants_image/0013035.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)
writer.close()

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normalize", img_norm, 1)
writer.close()

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
writer.close()
print(img_resize.size)

# Compose
trans_resize_2 = transforms.Resize(512)
# Compose参数中前面一个的输出，是后面一个的输入。
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
writer.close()

# RandomCrop
trans_randomcrop = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)
