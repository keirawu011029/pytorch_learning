from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter


# 通过transforms.ToTensor()看两个问题
# 1. transforms该如何使用
# 2. 为什么我们需要Tensor数据类型

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("./logs")

# 创建一个工具
img_trans = transforms.ToTensor()
# 使用工具，输入： 输出：
img_tensor = img_trans(img)

writer.add_image("img_tensor", img_tensor)

writer.close()

# cv_img = cv2.imread(img_path)