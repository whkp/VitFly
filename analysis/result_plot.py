#详细的统计分析

# 1. **碰撞对比** (`collision_comparison.png`)
#    - 试验碰撞率
#    - 成功率对比

# 2. **轨迹对比** (`trajectory_comparison.png`)
#    - XY和XZ平面的轨迹散点图对比

# 3. **3D轨迹对比** (`trajectory_3d_comparison.png`)
#    - 三维空间的模型轨迹对比

# 4. **性能摘要表** (`summary_table.png`, `model_comparison_summary.csv`)
#    - 详细的模型性能指标表格

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from os.path import join as opj
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D
import json

class ModelConfig:
    """Configuration class for model visualization settings."""
    
    def __init__(self, name, color, marker='o', linestyle='-', folder_name=None):
        self.name = name
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.folder_name = folder_name or name.lower().replace('robust', '').replace('+', '').replace(' ', '')

class TrajectoryDataLoader:
    """
    Load and preprocess trajectory data from CSV files for different models.
    """
    
    def __init__(self, data_folder, model_config):
        """
        Initialize the data loader for a specific model.
        
        Args:
            data_folder (str): Path to the folder containing numbered subfolders with data.csv files
            model_config (ModelConfig): Configuration object for the model
        """
        self.model_config = model_config
        self.model_name = model_config.name
        self.data_folder = data_folder
        self.trajectory_data = []
        self.collision_data = []
        
        self._load_data()
    
    def _load_data(self):
        """Load trajectory data from all subfolders."""
        # Get all numbered subfolders
        subfolders = sorted([d for d in os.listdir(self.data_folder) 
                           if os.path.isdir(opj(self.data_folder, d)) and d.isdigit()])
        
        if not subfolders:
            print(f"No numbered subfolders found in {self.data_folder}")
            return
        
        print(f"Loading data for {self.model_name} from {len(subfolders)} subfolders")
        
        for subfolder in subfolders:
            csv_path = opj(self.data_folder, subfolder, "data.csv")
            if os.path.exists(csv_path):
                try:
                    # Read CSV data
                    df = pd.read_csv(csv_path)
                    
                    # Store the dataframe for this trial
                    self.trajectory_data.append(df)
                    
                    # Extract collision information if available
                    if 'is_collide' in df.columns:
                        self.collision_data.append(df['is_collide'].values)
                    else:
                        # If no collision column, assume no collisions
                        self.collision_data.append(np.zeros(len(df), dtype=bool))
                
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    continue
        
        print(f"Successfully loaded {len(self.trajectory_data)} trials for {self.model_name}")

    def get_collision_statistics(self):
        """Calculate collision statistics across all trials."""
        if not self.collision_data:
            return {}
        
        total_trials = len(self.collision_data)
        trials_with_collisions = sum(1 for collision_array in self.collision_data 
                                   if np.any(collision_array))
        collision_rate = trials_with_collisions / total_trials if total_trials > 0 else 0
        
        # Total collision timesteps across all trials
        total_collision_timesteps = sum(np.sum(collision_array) for collision_array in self.collision_data)
        total_timesteps = sum(len(collision_array) for collision_array in self.collision_data)
        timestep_collision_rate = total_collision_timesteps / total_timesteps if total_timesteps > 0 else 0
        
        return {
            'total_trials': total_trials,
            'trials_with_collisions': trials_with_collisions,
            'trial_collision_rate': collision_rate,
            'timestep_collision_rate': timestep_collision_rate,
            'success_rate': 1 - collision_rate
        }

    def get_position_statistics(self, speed_filter=None):
        """
        Get position statistics for trajectory analysis.
        
        Args:
            speed_filter (float): Filter trials by desired velocity (optional)
        """
        if not self.trajectory_data:
            return None
        
        all_positions = []
        
        for df in self.trajectory_data:
            # Filter by speed if specified
            if speed_filter is not None and 'desired_vel' in df.columns:
                speed_data = df[np.abs(df['desired_vel'] - speed_filter) < 0.5]
                if len(speed_data) == 0:
                    continue
                df = speed_data
            
            # Extract position data
            if all(['pos_x' in df.columns, 'pos_y' in df.columns, 'pos_z' in df.columns]):
                positions = df[['pos_x', 'pos_y', 'pos_z']].values
                all_positions.append(positions)
        
        if not all_positions:
            return None
        
        # Concatenate all position data
        all_pos = np.concatenate(all_positions, axis=0)
        
        return {
            'positions': all_pos,
            'mean_x': np.mean(all_pos[:, 0]),
            'mean_y': np.mean(all_pos[:, 1]),
            'mean_z': np.mean(all_pos[:, 2]),
            'std_x': np.std(all_pos[:, 0]),
            'std_y': np.std(all_pos[:, 1]),
            'std_z': np.std(all_pos[:, 2])
        }


class ComparisonPlotter:
    """Class for creating comparison plots between   different models."""
    
    def __init__(self, data_loaders, output_dir):
        """
        Initialize the plotter with data loaders for different models.
        
        Args:
            data_loaders (dict): Dictionary of model_name -> TrajectoryDataLoader
            output_dir (str): Directory to save plots
        """
        self.data_loaders = data_loaders
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_collision_comparison(self):
        """Create collision rate comparison plot."""
        model_names = []
        trial_collision_rates = []
        timestep_collision_rates = []
        success_rates = []
        colors = []
        
        for model_name, loader in self.data_loaders.items():
            stats = loader.get_collision_statistics()
            if stats:
                model_names.append(loader.model_config.name)
                trial_collision_rates.append(stats['trial_collision_rate'] * 100)
                timestep_collision_rates.append(stats['timestep_collision_rate'] * 100)
                success_rates.append(stats['success_rate'] * 100)
                colors.append(loader.model_config.color)
        
        if not model_names:
            print("No collision data available for comparison")
            return
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trial-based collision rates
        bars1 = ax1.bar(model_names, trial_collision_rates, color=colors, alpha=0.7)
        ax1.set_ylabel('Trial Collision Rate (%)')
        ax1.set_title('Collision Rate by Trial')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, trial_collision_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Success rates
        bars2 = ax2.bar(model_names, success_rates, color=colors, alpha=0.7)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate by Model')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(opj(self.output_dir, 'collision_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'collision_comparison.pdf'), bbox_inches='tight')
        plt.close()
    
    def create_trajectory_comparison(self):
        """Create trajectory path comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for model_name, loader in self.data_loaders.items():
            pos_stats = loader.get_position_statistics()
            if pos_stats is not None:
                positions = pos_stats['positions']
                
                # Sample some trajectories for visualization (to avoid overcrowding)
                if len(positions) > 1000:
                    sample_indices = np.random.choice(len(positions), 1000, replace=False)
                    sample_positions = positions[sample_indices]
                else:
                    sample_positions = positions
                
                # XY plot
                ax1.scatter(sample_positions[:, 0], sample_positions[:, 1], 
                           c=loader.model_config.color, alpha=0.3, s=1,
                           label=loader.model_config.name)
                
                # XZ plot
                ax2.scatter(sample_positions[:, 0], sample_positions[:, 2], 
                           c=loader.model_config.color, alpha=0.3, s=1,
                           label=loader.model_config.name)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Trajectory Comparison - XY Plane')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('Trajectory Comparison - XZ Plane')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(opj(self.output_dir, 'trajectory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'trajectory_comparison.pdf'), bbox_inches='tight')
        plt.close()
    
    def create_3d_trajectory_plot(self):
        """Create 3D trajectory visualization."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        for model_name, loader in self.data_loaders.items():
            pos_stats = loader.get_position_statistics()
            if pos_stats is not None:
                positions = pos_stats['positions']
                
                # Sample positions for 3D visualization
                if len(positions) > 500:
                    sample_indices = np.random.choice(len(positions), 500, replace=False)
                    sample_positions = positions[sample_indices]
                else:
                    sample_positions = positions
                
                ax.scatter(sample_positions[:, 0], sample_positions[:, 1], sample_positions[:, 2],
                          c=loader.model_config.color, alpha=0.4, s=2,
                          label=loader.model_config.name)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Trajectory Comparison')
        ax.legend()
        
        # Set reasonable axis limits
        ax.set_xlim([0, 60])
        ax.set_ylim([-5, 5])
        ax.set_zlim([0, 5])
        
        plt.tight_layout()
        plt.savefig(opj(self.output_dir, 'trajectory_3d_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'trajectory_3d_comparison.pdf'), bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self):
        """Create a summary table of all metrics."""
        summary_data = []
        
        for model_name, loader in self.data_loaders.items():
            collision_stats = loader.get_collision_statistics()
            pos_stats = loader.get_position_statistics()
            
            if collision_stats and pos_stats:
                summary_data.append({
                    'Model': loader.model_config.name,
                    'Total Trials': collision_stats['total_trials'],
                    'Success Rate (%)': collision_stats['success_rate'] * 100,
                    'Trial Collision Rate (%)': collision_stats['trial_collision_rate'] * 100,
                    'Timestep Collision Rate (%)': collision_stats['timestep_collision_rate'] * 100,
                    'Avg Y Position': pos_stats['mean_y'],
                    'Avg Z Position': pos_stats['mean_z'],
                    'Y Std Dev': pos_stats['std_y'],
                    'Z Std Dev': pos_stats['std_z']
                })
        
        if not summary_data:
            print("No data available for summary table")
            return
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df = df.round(3)
        df.to_csv(opj(self.output_dir, 'model_comparison_summary.csv'), index=False)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Select key metrics for display
        display_cols = ['Model', 'Success Rate (%)', 'Trial Collision Rate (%)', 'Y Std Dev', 'Z Std Dev']
        display_df = df[display_cols]
        
        table = ax.table(cellText=display_df.values,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # Model name column
                    cell.set_facecolor('#E8F5E8')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F5F5F5')
        
        plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        plt.savefig(opj(self.output_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'summary_table.pdf'), bbox_inches='tight')
        plt.close()
        
        print("\nModel Comparison Summary:")
        print(df.to_string(index=False))


def analyze_model_comparison(data_directory, output_dir='./analysis_plots'):
    """
    Main function to perform comprehensive model comparison analysis.
    
    Args:
        data_directory (str): Path to the data directory containing model folders
        output_dir (str): Directory to save analysis plots
    """
    
    # Define model configurations based on actual folder structure
    model_configs = {
        'expert': ModelConfig('Expert', 'saddlebrown', 'x', '--', 'expert'),
        'vit': ModelConfig('ViT', 'gray', 's', '-', 'vit'),
        'vitlstm': ModelConfig('ViT+LSTM', 'green', 'o', '-', 'vitlstm'),
        'convnet': ModelConfig('ConvNet', 'steelblue', '^', '-', 'convnet'),
        'robustVitLstm': ModelConfig('RobustViT+LSTM', 'red', 'v', '-', 'robustVitLstm')
    }
    
    # Initialize data loaders
    data_loaders = {}
    
    print("Loading trajectory data for model comparison...")
    print("=" * 60)
    
    for key, config in model_configs.items():
        model_path = opj(data_directory, config.folder_name)
        
        if os.path.exists(model_path):
            loader = TrajectoryDataLoader(model_path, config)
            if loader.trajectory_data:  # Check if data was loaded successfully
                data_loaders[key] = loader
                print(f"✓ {config.name}: {len(loader.trajectory_data)} trials loaded")
            else:
                print(f"✗ {config.name}: No valid data found")
        else:
            print(f"✗ {config.name}: Directory not found at {model_path}")
    
    if not data_loaders:
        print("No valid data found for any model!")
        return
    
    print(f"\nGenerating comparison plots for {len(data_loaders)} models...")
    
    # Initialize comparison plotter
    plotter = ComparisonPlotter(data_loaders, output_dir)
    
    # Generate all comparison plots
    print("- Creating collision comparison plots...")
    plotter.create_collision_comparison()
    
    print("- Creating trajectory comparison plots...")
    plotter.create_trajectory_comparison()
    
    print("- Creating 3D trajectory visualization...")
    plotter.create_3d_trajectory_plot()
    
    print("- Generating summary table...")
    plotter.create_summary_table()
    
    print(f"\nAll comparison plots saved to {output_dir}/")
    print("Analysis complete!")


# Main execution
if __name__ == "__main__":
    # Set the data directory path
    data_directory = "./data"
    output_directory = "./analysis_plots"
    
    print("Starting model comparison analysis...")
    print("=" * 50)
    
    # Run the comprehensive analysis
    analyze_model_comparison(data_directory, output_directory)
    
    print("\nAnalysis complete! Check the 'analysis_plots' directory for all visualizations.")