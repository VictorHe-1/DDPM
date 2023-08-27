from data.diffusion_dataset import MnistDataset
from configs.base_config import Configs


if __name__ == "__main__":
    configs = Configs()
    configs.update({
                       'dataset': MnistDataset(image_size=configs.image_size, data_path="./data"),
                       'image_channels': 1,
                       'epochs': 5})
    configs.init()
    configs.run()
