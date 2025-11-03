import torch


def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {avg_loss:.4f}')
    
    return accuracy, avg_loss