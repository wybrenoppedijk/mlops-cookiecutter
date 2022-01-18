import torch
import torchvision
import torchdrift
import hydra
import sklearn
import matplotlib.pyplot as plt
from src.models.model import MyAwesomeModel
import hydra

from src.data.dataset import MNIST_Corrupted

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main(config):
    corrupt = torchvision.transforms.Compose([
    torchvision.transforms.GaussianBlur(3),
    torchvision.transforms.ToTensor()])

    train_dataset = MNIST_Corrupted(hydra.utils.get_original_cwd() +'/data/processed/corruptmnist', train=True)
    val_dataset = MNIST_Corrupted(hydra.utils.get_original_cwd() +'/data/processed/corruptmnist', train=False, transform=corrupt)
    # torch dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    train_loader_odd = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
    model = MyAwesomeModel.load_from_checkpoint(hydra.utils.get_original_cwd() + "/model_store/corrupted_mnist.ckpt", config=config)
    feature_extractor = model.conv
    feature_extractor.fc = torch.nn.Identity()
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(train_loader,feature_extractor, drift_detector)

    drift_detection_model = torch.nn.Sequential(
        feature_extractor,
        drift_detector
    )

    inputs = next(iter(train_loader))[0]
    features = feature_extractor(inputs)
    score = drift_detector(features.view(-1,features.shape[1]))
    p_val = drift_detector.compute_p_value(features.view(-1,features.shape[1]))
    score, p_val

    N_base = drift_detector.base_outputs.size(0)
    mapper = sklearn.manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features)
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f'score {score:.2f} p-value {p_val:.2f}');

    inputs_ood = next(iter(train_loader_odd))
    features = feature_extractor(inputs_ood)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)

    features_embedded = mapper.transform(features)
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f'score {score:.2f} p-value {p_val:.2f}')

if __name__ == "__main__":
    main()
