import json
import os
from datetime import datetime
import pandas as pd

class Saver:
    ''' This class is meant to save all of the data created by the training phase and load experiments
    as well'''
    # Create the directory where the experiment will be saved in
    def __init__(self, dir='./experiments'):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get the name for the experiment
    def create_dir_exp(self, model_name, config):
        'Make directory for a model'
        exp_name = f"{model_name}_{config['batch_size']}_dr{config['dropout_rate']}_aug{config['augmentation']}_{self.timestamp}"
        exp_dir = os.path.join(self.dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    # Save the configuration of the model
    def save_config(self, exp_dir, config):
        """Save configuration of experiment"""
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    # Save all the history of the trainingg phase
    def save_history(self, exp_dir, history):
        """Save history of experiment"""
        with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)

    # Save the metrics 
    def save_metrics(self, exp_dir, metrics):
        """Save metrics of model"""
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_summary(self, exp_dir, summary):
        """Save summary of experiment"""
        with open(os.path.join(exp_dir, 'summary.txt'), 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    # Load the data of an experiment
    def load_experiment(self, exp_dir):
        """Load a whole experiment"""
        config = json.load(open(os.path.join(exp_dir, 'config.json')))
        history = json.load(open(os.path.join(exp_dir, 'history.json')))
        metrics = json.load(open(os.path.join(exp_dir, 'metrics.json')))
        return config, history, metrics

    def create_comparison_table(self, experiments_list):
        """Create comparative table of multiple experiments"""
        data = []
        for exp_dir in experiments_list:
            config, history, metrics = self.load_experiment(exp_dir)
            data.append({
                'Model': config['model_name'],
                'Batch Size': config['batch_size'],
                'Dropout': config['dropout_rate'],
                'Augmentation': config['augmentation'],
                'Params': metrics['total_parameters'],
                'Test Acc': metrics['test_accuracy'],
                'Train Time': metrics['total_training_time'],
                'Inference (samples/s)': metrics['inference_speed']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.dir, 'comparison_table.csv'), index=False)
        return df