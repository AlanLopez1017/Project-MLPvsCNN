from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch
from sklearn.model_selection import train_test_split

def data_to_loader(batch_size, augmentation, model_name):
    """
    Create Dataloaders for CIFAR-10 with data augmentation or no.
    The dataset is loaded from torchvision
    """

    # Normalization (not applied)
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )

    # Data agmentation based on model type
    if augmentation:
        if model_name == 'CNN':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.RandomCrop(32, padding=4), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ###
                transforms.RandomRotation(15), ###
                transforms.ToTensor()
            ])
        else: # MLP
            train_transform= transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor()
            ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR dataset from torchvision
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    class_names = train_dataset.classes

    # train labels
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

    # Division for training and validation indices
    train_indices, val_indices = train_test_split(
    range(len(train_dataset)),
    test_size=0.2,
    stratify=train_labels,
    random_state=42
    )
    
    train_dataset = Subset(datasets.CIFAR10(root='./data', train=True, download = True, transform=train_transform), train_indices)
    val_dataset = Subset(datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform),val_indices)

    # to dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)#, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#, num_workers=2)

    print(f"\n CIFAR-10 Dataset ")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_names
