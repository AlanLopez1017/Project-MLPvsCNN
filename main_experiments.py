import torch
import torchvision
import os
import torchvision.transforms as transforms
from setup import *
from MLP import *
from MLP_2 import *
from CNN import *
from CNN_2 import *
from training import train
from evaluate_model_and_metrics import *
from plot_cm import plot_cm
from learning_curves import learning_curves
from inference_time import inference_time
from data_to_loader import data_to_loader
from Saver import *
from ParameterMatchingExperiment import *

def run_experiment(config, saver):
    print(f"Experiment: {config['model_name']}\n")
    print(f"Configuration: {config}")

    # Create directory of experiment and save configuration
    exp_dir = saver.create_dir_exp(config['model_name'], config)
    saver.save_config(exp_dir, config)
    
    # Create each loader for the data and extract the class names
    train_loader, val_loader, test_loader, class_names = data_to_loader(
        config['batch_size'], config['augmentation'], config['model_name']
    )

    # Create the model based on the model name
    if config['model_name'] == 'MLP':
        device, model, criterion, optimizer, model_info = create_model(
            MLP, config['input_size'], config['hidden_sizes'], 
            config['output_size'], config['dropout_rate'], config['model_name'], is_cpu = config['is_cpu']
        )
    elif config['model_name'] == 'CNN':
        device, model, criterion, optimizer, model_info = create_model(
            model_class = CNN, output_size = config['output_size'], 
            dropout_rate = config['dropout_rate'], model_name = config['model_name'], is_cpu = config['is_cpu']
        )
    elif config['model_name'] == 'MLP_820': # MLP with 820k parameters
        device, model, criterion, optimizer, model_info = create_model(
            model_class = MLP_820, model_name = config['model_name'], is_cpu = config['is_cpu'])
    elif config['model_name'] == 'CNN_820': # CNN with 820k parameters
        device, model, criterion, optimizer, model_info = create_model(
            model_class = CNN_820, model_name = config['model_name'], is_cpu = config['is_cpu'])

    # Training the model
    save_path = os.path.join(exp_dir, 'best_model.pth') # Save path for the best model
    history = train(model, device, optimizer, criterion, train_loader, val_loader, 
                    config['epochs'], save_path, early_stopping=True)
    saver.save_history(exp_dir, history) # Save history of the training

    # Load best model
    model.load_state_dict(torch.load(save_path)) # Load the best model

    # Evaluate the model
    cm, test_metrics = evaluate_model_and_metrics(model, test_loader, device, class_names, criterion)
    
    # all metrics
    all_metrics = {**model_info, **test_metrics, **history}
    saver.save_metrics(exp_dir, all_metrics)

    # Visualizations
    plot_cm(cm, config['model_name'], class_names, os.path.join(exp_dir, 'confusion_matrix.png')) 
    learning_curves(history, os.path.join(exp_dir, 'learning_curves.png')) # Plot learning curves (loss and accuracy vs epochs of the training and validation)
    inference_time(model, test_loader, device) # Calculate the inference time of the model

    summary = {
        'Model': config['model_name'],
        'Total trainable parameters' : model_info['trainable_parameters'],
        'Total Parameters': model_info['total_parameters'],
        'Test Accuracy': f"{test_metrics['test_accuracy']:.2f}%",
        'Training Time': f"{history['total_training_time']:.2f}s",
        'Inference Speed': f"{test_metrics['inference_speed']:.2f} samples/s",
        'Best Val Accuracy': f"{history['best_val_acc']:.2f}%"
    }
    saver.save_summary(exp_dir, summary) # Save summary of the experiment

    return exp_dir, all_metrics # Return the directory of the experiment and the metrics

def run():
    saver = Saver(dir='./asd') 

    # Base configuration for the experiments
    base_config = {
        'input_size': 3072,
        'hidden_sizes': [512, 256, 128],
        'output_size': 10,
        'epochs': 50
    }

    # configuration for the experiments
    experiments = [
        #{**base_config, 'model_name': 'MLP', 'batch_size': 128, 'dropout_rate': 0.0, 'augmentation': False, 'is_cpu': False}, # No augmentation
        #{'output_size': 10,'epochs': 50,'model_name': 'CNN', 'batch_size': 128, 'dropout_rate': 0.0, 'augmentation': False, 'is_cpu': False}
        #{**base_config, 'model_name': 'MLP', 'batch_size': 128, 'dropout_rate': 0.0, 'augmentation': True, 'is_cpu': False}, # Augmentation
        #{'output_size': 10,'epochs': 50,'model_name': 'CNN', 'batch_size': 128, 'dropout_rate': 0.0, 'augmentation': True, 'is_cpu': False}
        #{**base_config, 'model_name': 'MLP', 'batch_size': 128, 'dropout_rate': 0.3, 'augmentation': True, 'is_cpu': False}, # Dropout
        #{'output_size': 10,'epochs': 50,'model_name': 'CNN', 'batch_size': 128, 'dropout_rate': 0.3, 'augmentation': True, 'is_cpu': False}
        #{'output_size': 10,'epochs': 50,'model_name': 'MLP_820', 'batch_size': 128, 'dropout_rate': 0.0, 'augmentation': True, 'is_cpu': False}, # MLP 820k
        {'output_size': 10,'epochs': 50,'model_name': 'CNN_820', 'batch_size': 128, 'dropout_rate': 0.0, 'augmentation': True, 'is_cpu': False} # CNN 820k
        #{'output_size': 10,'epochs': 50,'model_name': 'MLP_820', 'batch_size': 128, 'dropout_rate': 0.3, 'augmentation': True, 'is_cpu': False}, # MLP 820k dropout
        #{'output_size': 10,'epochs': 50,'model_name': 'CNN_820', 'batch_size': 128, 'dropout_rate': 0.3, 'augmentation': True, 'is_cpu': False} # CNN 820k dropout
        #{'output_size': 10,'epochs': 50,'model_name': 'CNN_820', 'batch_size': 128, 'dropout_rate': 0.3, 'augmentation': False, 'is_cpu': False} # CPU 
    ]


    experiment_dirs = []
    all_results = []

    # Run the experiments
    for i, config in enumerate(experiments, 1):

        print(f'Experiment {i}\n')
    
        dir, metrics = run_experiment(config, saver)
        experiment_dirs.append(dir)
        all_results.append(metrics)

    comparison_df = saver.create_comparison_table(experiment_dirs)

    return experiment_dirs, all_results, comparison_df

if __name__ == '__main__':
    experiment_dirs, all_results, comparison_df = run()
