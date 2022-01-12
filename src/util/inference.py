import torchvision

from torchvision import transforms
import time

from src.data.lfw_dataset import LFWDataset

lfw_trans = transforms.Compose(
    [transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()]
)

# Define dataset
dataloader = LFWDataset('data/raw/lfw', lfw_trans)
imgs = next(iter(dataloader))
large = torchvision.models.mobilenet_v3_large(pretrained=True)
resnet = torchvision.models.resnet152(pretrained=True)

start_mobile = time.time()
a = large.forward(imgs)
end_mobile = time.time()

start_resnet = time.time()
b = resnet.forward(imgs)
end_resnet = time.time()

print(f"MobileNetV3 took {end_mobile - start_mobile} seconds"
      f"\nResNet took {end_resnet - start_resnet} seconds")

