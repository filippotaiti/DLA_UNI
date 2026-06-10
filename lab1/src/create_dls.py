import torchvision.transforms.v2 as T 
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import GTSRB

def create_dataloaders(training_set_perc, seed, batch_size, num_workers):

    transform = T.Compose([T.Resize(70), T.RandomCrop((64, 64)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gen = torch.Generator().manual_seed(seed)

    ds_train = GTSRB('_data/', split='train', transform=transform, download=True)
    test_set  = GTSRB('_data/', split='test', transform=transform, download=True)

    train_size = int(training_set_perc*len(ds_train))
    val_size = len(ds_train) - train_size

    training_set, validation_set = random_split(
        dataset=ds_train,
        lengths=[train_size, val_size],
        generator=gen

    )

    train_dl = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl