import torchvision


class MnistDataset(torchvision.datasets.MNIST):
    def __init__(self, image_size, data_path):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(image_size),
             torchvision.transforms.ToTensor()]
        )
        super().__init__(data_path, train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]