import numpy as np
import torch 
import time
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def evaluate_model_and_metrics(model, test_loader, device, class_names, criterion):
    ''' This function makes the prediction of the model and calculate the inference time,
    the global metrics and per classes'''
    model.eval()
    all_predictions = []
    all_labels = []
    test_loss = 0
  
    inference_start = time.time()

    # Evaluate mode
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    inference_time = time.time() - inference_start
    
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate the metrics
    print('Metrics per class')
    report = classification_report(all_labels, all_predictions, target_names = class_names, digits = 4)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average = None)

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")

    print(f"\n{'Macro Avg':<15} {precision.mean():<12.4f} {recall.mean():<12.4f} {f1.mean():<12.4f}")

    # Calculate the test accuracy
    accuracy = (all_predictions == all_labels).sum() / len(all_labels)
    avg_test_loss = test_loss / len(test_loader)

    print(f'\nTest accuracy: {accuracy * 100: .2f}%')
    print(f'Test Loss: {avg_test_loss:.4f}')

    # Get the confusion matrix of the model
    cm = confusion_matrix(all_labels, all_predictions)

    metrics = {
        'test_accuracy': accuracy,
        'test_loss': avg_test_loss,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'macro_precision': float(precision.mean()),
        'macro_recall': float(recall.mean()),
        'macro_f1': float(f1.mean()),
        'inference_time': inference_time,
        'inference_speed': len(all_labels) / inference_time,
        'confusion_matrix': cm.tolist()
    }
    
    return cm, metrics
    

