import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

def to_loader(train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size, augmentation, model_name):

    # Define traonsformation of data augmentation
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # --- Datasets ---
    if augmentation:
        # Usa el dataset personalizado con augmentations
        train_dataset = AugmentedDataset(train_data, train_labels, model_name, transform=train_transform)
    else:
        if model_name == 'CNN':
            train_data_t = torch.FloatTensor(train_data.reshape(-1, 3, 32, 32))
        else:
            train_data_t = torch.FloatTensor(train_data)
        train_labels_t = torch.LongTensor(train_labels)
        train_dataset = TensorDataset(train_data_t, train_labels_t)

    # Validación y test sin augmentations
    if model_name == 'CNN':
        val_data_t = torch.FloatTensor(val_data.reshape(-1, 3, 32, 32))
        test_data_t = torch.FloatTensor(test_data.reshape(-1, 3, 32, 32))
    else:
        val_data_t = torch.FloatTensor(val_data)
        test_data_t = torch.FloatTensor(test_data)

    val_labels_t = torch.LongTensor(val_labels)
    test_labels_t = torch.LongTensor(test_labels)

    val_dataset = TensorDataset(val_data_t, val_labels_t)
    test_dataset = TensorDataset(test_data_t, test_labels_t)

    # DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'\n=== Creating batches ===')
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def create_model(model_class = 'MLP', input_size = 512, hidden_sizes = 512, output_size = 10, dropout_rate = 0.0, model_name = 'MLP', is_cpu = False):
    
    ''' Create the model given the model name, calculate the number of trainable parameters and the criterion and optimizer
    of the model as well. '''
    print(f'\n=== Creating model {model_name} ===')
    
    device = torch.device("cpu" if is_cpu else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    # Creating model
    if model_name == 'MLP':
        model = model_class(input_size, hidden_sizes, output_size, dropout_rate).to(device)
    elif model_name == 'CNN':
        model = model_class(output_size, dropout_rate).to(device)
    elif model_name == 'MLP_820':
        model = model_class().to(device)
    elif model_name == 'CNN_820':
        model = model_class().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Model summary
    trainable, total = parameters(model, model_name)

    model_info = {
        'trainable_parameters' : trainable,
        'total_parameters' : total
    }

    return device, model, criterion, optimizer, model_info


def parameters(model, model_name):
    '''Amount of parameters of the model'''
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    parameters = trainable + non_trainable

    print(f'\n== Model parameters of {model_name}==')
    print(f'Trainable parameters: {trainable:,}')
    print(f'Non-trainable parameters: {non_trainable:,}')
    print(f'Total parameters of the model: {parameters:,}')

    return trainable, parameters

class AugmentedDataset(Dataset):
    """Dataset para aplicar data augmentation a imágenes vectorizadas (para MLP)."""
    def __init__(self, data, labels, model_name, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32)  # (3072,) → (3,32,32)
        img = img.transpose(1, 2, 0)             # → (32,32,3)
        label = self.labels[idx]

        img = (img * 255).astype('uint8')        # restaurar rango 0–255
        img = transforms.ToPILImage()(img)       # convertir a imagen PIL

        if self.transform:
            img = self.transform(img)

        if self.model_name == 'MLP':
            img = img.view(-1)                       # volver a vector (3072,)
        return img, label

