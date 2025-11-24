import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def extract_data_from_latex_table(latex_file_path):
    """
    Extract data from LaTeX table in result.tex
    Only extract Original Dataset data (not PCA or Statistical Features)
    
    Returns:
        dict: Dictionary with structure:
              {
                  'Contact': {'PatchTST': {'F1': ..., 'Precision': ..., 'Recall': ...}, ...},
                  'Ring': {...},
                  'PCB': {...}
              }
    """
    with open(latex_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Initialize data structure
    data = {
        'Contact': {},
        'Ring': {},
        'PCB': {}
    }
    
    current_model = None
    
    # Process line by line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines and LaTeX commands
        if not line or line.startswith('%'):
            continue
        
        # Check if line contains multirow (model name)
        multirow_match = re.search(r'\\multirow\{3\}\{\*\}\{([^}]+)\}', line)
        if multirow_match:
            current_model = multirow_match.group(1)
        
        # Check if line contains data (has & and numbers)
        if '&' in line and current_model:
            # Try to extract data
            # Remove LaTeX commands but keep &
            cleaned_line = line.replace('\\\\', '').strip()
            
            # Split by &
            parts = [p.strip() for p in cleaned_line.split('&')]
            
            # Look for dataset name
            dataset = None
            for part in parts:
                if 'Contact' in part:
                    dataset = 'Contact'
                    break
                elif 'Ring' in part:
                    dataset = 'Ring'
                    break
                elif 'PCB' in part:
                    dataset = 'PCB'
                    break
            
            # If we found a dataset, extract the numeric values
            if dataset:
                # Extract all numeric values from the line
                numbers = []
                for part in parts:
                    # Remove any LaTeX commands
                    cleaned = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', part)
                    cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned).strip()
                    
                    # Try to convert to float
                    try:
                        num = float(cleaned)
                        numbers.append(num)
                    except ValueError:
                        continue
                
                # Extract only the first 3 numbers (Original Dataset: F1, Precision, Recall)
                # Both tables have the same structure: first 3 numbers are from Original Dataset
                if len(numbers) >= 3:
                    data[dataset][current_model] = {
                        'F1': numbers[0],
                        'Precision': numbers[1],
                        'Recall': numbers[2]
                    }
    
    return data


def plot_metrics_by_dataset(data, output_dir):
    """
    Plot F1, Precision, and Recall for each dataset (Original Dataset only)
    Models are sorted by F1 score from high to low
    
    Args:
        data: Dictionary with structure from extract_data_from_latex_table
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define colors and markers
    colors = {'F1': '#2E86AB', 'Precision': '#A23B72', 'Recall': '#F18F01'}
    markers = {'F1': 'o', 'Precision': 's', 'Recall': '^'}
    
    # Plot for each dataset
    for dataset_name, dataset_data in data.items():
        if not dataset_data:
            continue
        
        # Sort models by F1 score (high to low)
        models_with_f1 = [(model, values['F1']) for model, values in dataset_data.items()]
        models_with_f1.sort(key=lambda x: x[1], reverse=True)
        models = [model for model, _ in models_with_f1]
        
        # Create figure with single plot (Original Dataset only)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'{dataset_name} Dataset - Model Performance Comparison (Original Dataset)', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Extract metrics
        metrics = ['F1', 'Precision', 'Recall']
        
        # Plot each metric
        x_positions = np.arange(len(models))
        
        for metric in metrics:
            values = [dataset_data[model][metric] for model in models]
            
            # Plot line
            ax.plot(x_positions, values, 
                   color=colors[metric], 
                   marker=markers[metric],
                   linewidth=2.5,
                   markersize=10,
                   label=metric,
                   alpha=0.8)
            
            # Add value labels on data points
            for i, (x, y) in enumerate(zip(x_positions, values)):
                ax.annotate(f'{y:.4f}', 
                           xy=(x, y), 
                           xytext=(0, 8),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           fontweight='bold',
                           color=colors[metric],
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=colors[metric],
                                   alpha=0.7))
        
        # Customize plot
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        # Add minor gridlines
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')
        
        plt.tight_layout()
        
        # Save figure
        output_file = output_path / f'{dataset_name.lower()}_metrics_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'Saved plot: {output_file}')
        plt.close()


def plot_all_metrics_single_plot(data, output_dir):
    """
    Plot all metrics for all datasets in a single figure with subplots
    Only shows Original Dataset, models sorted by F1 score
    
    Args:
        data: Dictionary with structure from extract_data_from_latex_table
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define colors and markers
    colors = {'F1': '#2E86AB', 'Precision': '#A23B72', 'Recall': '#F18F01'}
    markers = {'F1': 'o', 'Precision': 's', 'Recall': '^'}
    
    # Create figure with 3 subplots (3 datasets, Original only)
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle('Model Performance Comparison Across All Datasets (Original Dataset)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    dataset_names = ['Contact', 'Ring', 'PCB']
    
    for row_idx, dataset_name in enumerate(dataset_names):
        dataset_data = data.get(dataset_name, {})
        
        if not dataset_data:
            continue
        
        # Sort models by F1 score (high to low)
        models_with_f1 = [(model, values['F1']) for model, values in dataset_data.items()]
        models_with_f1.sort(key=lambda x: x[1], reverse=True)
        models = [model for model, _ in models_with_f1]
        
        x_positions = np.arange(len(models))
        metrics = ['F1', 'Precision', 'Recall']
        
        ax = axes[row_idx]
        
        # Plot each metric
        for metric in metrics:
            values = [dataset_data[model][metric] for model in models]
            
            # Plot line
            ax.plot(x_positions, values, 
                   color=colors[metric], 
                   marker=markers[metric],
                   linewidth=2.5,
                   markersize=10,
                   label=metric,
                   alpha=0.8)
            
            # Add value labels on data points
            for i, (x, y) in enumerate(zip(x_positions, values)):
                ax.annotate(f'{y:.4f}', 
                           xy=(x, y), 
                           xytext=(0, 8),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8,
                           fontweight='bold',
                           color=colors[metric],
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=colors[metric],
                                   alpha=0.7))
        
        # Customize subplot
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset_name} Dataset', fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        # Add minor gridlines
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'all_metrics_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved comprehensive plot: {output_file}')
    plt.close()


def main():
    """
    Main function to extract data from LaTeX table and generate plots
    """
    # Set paths
    latex_file = Path('generate_image_for_conclusion_in_latex/result.tex')
    output_dir = Path('generate_image_for_conclusion_in_latex')
    
    # Extract data from LaTeX table
    print('Extracting data from LaTeX table...')
    data = extract_data_from_latex_table(latex_file)
    
    # Print extracted data for verification
    print('\nExtracted data (Original Dataset only):')
    for dataset, models in data.items():
        print(f'\n{dataset}:')
        # Sort models by F1 score for display
        models_sorted = sorted(models.items(), key=lambda x: x[1]['F1'], reverse=True)
        for model, values in models_sorted:
            print(f'  {model}:')
            print(f'    F1={values["F1"]:.4f}, '
                  f'Precision={values["Precision"]:.4f}, '
                  f'Recall={values["Recall"]:.4f}')
    
    # Generate plots
    print('\n\nGenerating plots...')
    plot_metrics_by_dataset(data, output_dir)
    plot_all_metrics_single_plot(data, output_dir)
    
    print('\n✓ All plots generated successfully!')


if __name__ == '__main__':
    main()

