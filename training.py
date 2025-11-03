import time
import torch
import numpy as np

def train(model, device, optimizer, criterion, train_loader, val_loader, epochs, save_name, early_stopping = True):
    # Training and validation
    best_val_acc = 0
    bound = 10 # for the early stopping
    counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    initial_training_time = time.time()

    for epoch in range(epochs):
        start_time = time.time()

        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{batch_idx+1}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Epoch time
        epoch_time = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        print(f'Epoch [{epoch+1}/{epochs}] - '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | ' f'Time: {epoch_time:.2f}s')
        
        # Early stopping
        if early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                # Save the model
                torch.save(model.state_dict(), save_name)
                print(f' Best model saved! Val Acc: {val_acc:.2f}%')
            else:
                counter += 1

                if counter>=bound:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                    break

    total_training_time = time.time() - initial_training_time


    print(f'Total training time: {total_training_time:.2f} seconds')
    print(f"Average time per epoch: {np.mean(history['epoch_time']):.2f} seconds")
    print(f'\nBest Validation Accuracy: {best_val_acc:.2f}%')

    history['final_epoch'] = epoch + 1
    history['Average time per epoch'] = np.mean(history['epoch_time'])
    history['total_training_time'] = total_training_time
    history['best_val_acc'] = best_val_acc

    return history