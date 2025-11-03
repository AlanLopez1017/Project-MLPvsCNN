import matplotlib.pyplot as plt


def learning_curves(history, save_path = 'lc.png'):
    '''This function plots the loss and accuracy of the training and validation given the history of the model '''
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize = 14, fontweight = 'bold')
    ax1.set_xlabel('Epoch', fontsize = 12)
    ax1.set_ylabel('Loss', fontsize = 12)
    ax1.legend(fontsize = 12)
    ax1.grid(True, alpha = 0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize = 14, fontweight = 'bold')
    ax2.set_xlabel('Epoch', fontsize = 12)
    ax2.set_ylabel('Accuracy (%)', fontsize = 12)
    ax2.legend(fontsize = 12)
    ax2.grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    #plt.show()