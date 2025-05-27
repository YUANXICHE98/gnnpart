import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os
import glob

# Path to the TensorBoard logs
log_dir = 'autodl-tmp/models/route1_gnn_softmask/checkpoints/logs'

# Function to extract scalar data from TensorBoard logs
def extract_tensorboard_scalars(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    
    # Initialize dictionaries to store data
    data = {}
    
    for event_file in event_files:
        print(f"Processing file: {event_file}")
        try:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            
            # Get list of scalar tags (metrics like loss, accuracy, etc.)
            tags = ea.Tags()['scalars']
            
            # Extract scalar values for each tag
            for tag in tags:
                events = ea.Scalars(tag)
                if tag not in data:
                    data[tag] = {'step': [], 'value': []}
                
                for event in events:
                    data[tag]['step'].append(event.step)
                    data[tag]['value'].append(event.value)
        except Exception as e:
            print(f"Error processing {event_file}: {e}")
    
    return data

# Extract metrics from TensorBoard logs
data = extract_tensorboard_scalars(log_dir)

# Visualize the training metrics
if data:
    # Create a figure with subplots
    metrics = list(data.keys())
    n_metrics = len(metrics)
    
    if n_metrics > 0:
        fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        
        # Ensure axs is always a list/array
        if n_metrics == 1:
            axs = [axs]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            steps = data[metric]['step']
            values = data[metric]['value']
            
            # Sort by steps to ensure chronological order
            sorted_indices = np.argsort(steps)
            steps = [steps[j] for j in sorted_indices]
            values = [values[j] for j in sorted_indices]
            
            axs[i].plot(steps, values)
            axs[i].set_title(f'{metric} over Training Steps')
            axs[i].set_xlabel('Training Steps')
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        
        print(f"Metrics visualized: {metrics}")
        
        # Print final values for each metric
        print("\nFinal metric values:")
        for metric in metrics:
            if data[metric]['value']:
                final_value = data[metric]['value'][-1]
                print(f"{metric}: {final_value:.6f}")
    else:
        print("No metrics found in the logs.")
else:
    print("No data extracted from TensorBoard logs.") 