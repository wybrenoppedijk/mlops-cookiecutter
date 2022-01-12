import torchvision
from torch.utils.data import DataLoader
import ptflops
from torchvision import transforms
import time

from src.data.lfw_dataset import LFWDataset

def model_inference():
    lfw_trans = transforms.Compose(
        [transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()]
    )

    # Define dataset
    dataset = LFWDataset('data/raw/lfw', lfw_trans)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4
    )
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

    mobile_net = ptflops.get_model_complexity_info(large, print_per_layer_stat=True)
    resnet_flops = ptflops.get_model_complexity_info(resnet, print_per_layer_stat=True)

    print(f"MobileNetV3 FLOPS: {mobile_net}")
    print(f"ResNet FLOPS: {resnet_flops}")

if __name__ == "__main__":
    model_inference()