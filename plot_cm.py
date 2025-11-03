import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(cm, model_name, class_names, save_path = 'cm.png'):
    ''' Save the confusion matrix in a pretty way'''
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=class_names,
    yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()
