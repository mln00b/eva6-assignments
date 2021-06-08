from torchvision import datasets
import torch

from .transformations import get_train_test_transforms

def get_dataloaders(seed=42):
    train_transforms, test_transforms = get_train_test_transforms()

    train_ds = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_loader = torch.utils.data.DataLoader(train_ds, **dataloader_args)

    test_loader = torch.utils.data.DataLoader(test_ds, **dataloader_args)

    return train_loader, test_loader