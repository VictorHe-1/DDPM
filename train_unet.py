import argparse
from pathlib import Path

import torch
from torch import nn
import torchvision

from models.eps_model import UNet
from data.seg_dataset import CarvanaDataset


def train(data_loader, model, loss_func, optimizer, epochs, device):
    model.train()
    sigmoid = nn.Sigmoid()
    for epoch in range(epochs):
        for batch_idx, (image, mask) in enumerate(data_loader):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(image)
            mask = torchvision.transforms.functional.center_crop(mask, [logits.shape[2], logits.shape[3]])
            loss = loss_func(sigmoid(logits), mask)
            loss.backward()
            optimizer.step()
            if (batch_idx) % 30 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item()))


def create_model():
    pass

# python train_unet.py --data_path D:\pytorch_projects\DDPM\dataset\carvana-image-masking-challenge\train --mask_path D:\pytorch_projects\DDPM\dataset\carvana-image-masking-challenge\train_masks --epochs 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    batch_size = 1
    learning_rate = 2.5e-4
    device = torch.device("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")
    print(f"current device is {device}")

    model = UNet(in_channels=3, out_channels=1).to(device)
    # model = torch.compile(model)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = CarvanaDataset(Path(args.data_path), Path(args.mask_path))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)

    train(data_loader, model, loss_func, optimizer, args.epochs, device)