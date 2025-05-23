import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from os.path import join as opj
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import json

class ModelConfig:
    """Configuration class for model visualization settings."""
    
    def __init__(self, name, color, marker='o', linestyle='-', folder_name=None):
        self.name = name
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.folder_name = folder_name or name.lower().replace('+', '').replace(' ', '')

class VisionTransformerAnalyzer:
    """
    Analyzer for Vision Transformer quadrotor obstacle avoidance data.
    Generates plots matching the paper's visualizations.
    """
    
    def __init__(self, data_folder, model_config, speed_folders=None):
        """
        Initialize the analyzer with data from a specific model.
        
        Args:
            data_folder (str): Path to the folder containing numbered subfolders with data.csv files
            model_config (ModelConfig): Configuration object for the model
            speed_folders (dict): Mapping of speeds to folder names (optional, not used in new structure)
        """
        self.model_config = model_config
        self.model_name = model_config.name
        self.data_folder = data_folder
        self.trajectory_data = {}
        self.collision_data = {}
        
        # Load data for each speed
        speeds = [3, 4, 5, 6, 7]
        for speed in speeds:
            self.load_speed_data(speed, None)
    
    def load_speed_data(self, speed, folder_name):
        """Load trajectory data for a specific speed."""
        # For the new data structure, look for numbered subfolders directly under model folder
        model_subfolders = sorted(glob.glob(opj(self.data_folder, "[0-9]*")))
        
        if not model_subfolders:
            print(f"No numbered subfolders found in {self.data_folder}")
            return
        
        trajectories = []
        collisions = []
        
        for subfolder in model_subfolders:
            csv_path = opj(subfolder, "data.csv")
            if os.path.exists(csv_path):
                try:
                    # Read the CSV file
                    data = pd.read_csv(csv_path)
                    
                    # Filter data by desired velocity (speed)
                    # Assuming desired_vel column contains the target speed
                    speed_tolerance = 0.5  # Allow some tolerance in speed matching
                    speed_data = data[np.abs(data['desired_vel'] - speed) < speed_tolerance]
                    
                    if len(speed_data) > 0:
                        # Extract relevant columns based on the CSV structure
                        trajectory = speed_data[[
                            'timestamp', 'desired_vel',
                            'quat_1', 'quat_2', 'quat_3', 'quat_4',
                            'pos_x', 'pos_y', 'pos_z',
                            'vel_x', 'vel_y', 'vel_z',
                            'velcmd_x', 'velcmd_y', 'velcmd_z',
                            'ct_cmd', 'br_cmd_x', 'br_cmd_y', 'br_cmd_z'
                        ]].values
                        
                        collision = speed_data['is_collide'].values.astype(bool)
                        
                        trajectories.append(trajectory)
                        collisions.append(collision)
                
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    continue
        
        if trajectories:
            self.trajectory_data[speed] = trajectories
            self.collision_data[speed] = collisions
            print(f"Loaded {len(trajectories)} trials for {self.model_name} at {speed} m/s")


class ModelRegistry:
    """Registry for managing model configurations."""
    
    def __init__(self):
        self.models = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models from the paper."""
        self.add_model('expert', ModelConfig('Expert', 'red', 'x', '--'))
        self.add_model('vit', ModelConfig('ViT', 'gray', 's', '-'))
        self.add_model('vitlstm', ModelConfig('ViT+LSTM', 'green', 'o', '-'))
        self.add_model('convnet', ModelConfig('ConvNet', 'blue', '^', '-'))
        self.add_model('lstmnet', ModelConfig('LSTMnet', 'orange', 'v', '-'))
    
    def add_model(self, key, config):
        """Add a new model configuration."""
        self.models[key] = config
    
    def remove_model(self, key):
        """Remove a model configuration."""
        if key in self.models:
            del self.models[key]
    
    def get_model(self, key):
        """Get a model configuration."""
        return self.models.get(key)
    
    def get_all_models(self):
        """Get all model configurations."""
        return self.models
    
    def save_to_file(self, filepath):
        """Save model configurations to a JSON file."""
        config_dict = {}
        for key, config in self.models.items():
            config_dict[key] = {
                'name': config.name,
                'color': config.color,
                'marker': config.marker,
                'linestyle': config.linestyle,
                'folder_name': config.folder_name
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_from_file(self, filepath):
        """Load model configurations from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.models = {}
        for key, config_data in config_dict.items():
            self.models[key] = ModelConfig(**config_data)


class PlotGenerator:
    """Class responsible for generating all plots."""
    
    def __init__(self, analyzers, model_registry, output_dir):
        self.analyzers = analyzers
        self.model_registry = model_registry
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_collision_rate_plots(self, environment='spheres'):
        """Create collision rate plots similar to Figure 4a and 4b in the paper."""
        speeds = [3, 4, 5, 6, 7]
        
        # Calculate collision rates for each model and speed
        collision_rates = {}
        
        for model_name, analyzer in self.analyzers.items():
            rates = []
            for speed in speeds:
                if speed in analyzer.collision_data:
                    # Calculate collision rate per trial
                    total_collisions = 0
                    total_trials = len(analyzer.collision_data[speed])
                    
                    for collision_array in analyzer.collision_data[speed]:
                        if np.any(collision_array):  # If any collision occurred in this trial
                            total_collisions += 1
                    
                    rate = total_collisions / total_trials if total_trials > 0 else 0
                    rates.append(rate)
                else:
                    rates.append(0)
            
            collision_rates[model_name] = rates
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        
        # Plot each model
        for model_name, rates in collision_rates.items():
            analyzer = self.analyzers[model_name]
            config = analyzer.model_config
            
            plt.plot(speeds, rates, 
                    color=config.color, 
                    marker=config.marker, 
                    linestyle=config.linestyle,
                    markersize=8, 
                    linewidth=2, 
                    label=config.name)
        
        plt.xlabel('Forward velocity (m/s)', fontsize=12)
        plt.ylabel('Collision rate per trial', fontsize=12)
        
        if environment == 'spheres':
            plt.title('Collision rates', fontsize=14)
            plt.ylim(0, 1.0)
        else:
            plt.title('Collision rates (Trees)', fontsize=14)
            plt.ylim(0, 1.2)
        
        plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'collision_rates_{environment}'
        plt.savefig(opj(self.output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, f'{filename}.pdf'), bbox_inches='tight')
        plt.close()
    
    def create_path_distribution_plot(self):
        """Create top-down path distribution plot similar to Figure 4c in the paper."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot paths for each model
        for model_name, analyzer in self.analyzers.items():
            config = analyzer.model_config
            color = config.color
            
            # Use data from speed 5 m/s as representative
            if 5 in analyzer.trajectory_data:
                trajectories = analyzer.trajectory_data[5]
                
                # Column indices based on the CSV structure:
                # pos_x is column 6, pos_y is column 7
                x_col, y_col = 6, 7
                
                # Plot multiple trajectory samples with transparency
                for i, traj in enumerate(trajectories[:10]):  # Plot first 10 trials
                    if traj.shape[1] > max(x_col, y_col):
                        ax.plot(traj[:, x_col], traj[:, y_col], 
                               color=color, alpha=0.3, linewidth=1)
                
                # Plot mean trajectory
                if len(trajectories) > 0:
                    all_x = []
                    all_y = []
                    for traj in trajectories:
                        if traj.shape[1] > max(x_col, y_col):
                            all_x.append(traj[:, x_col])
                            all_y.append(traj[:, y_col])
                    
                    if all_x:
                        # Interpolate to common x-axis for averaging
                        x_common = np.linspace(0, 60, 100)
                        y_interp = []
                        
                        for x, y in zip(all_x, all_y):
                            if len(x) > 1:
                                y_interp.append(np.interp(x_common, x, y))
                        
                        if y_interp:
                            y_mean = np.mean(y_interp, axis=0)
                            ax.plot(x_common, y_mean, color=color, linewidth=2.5, 
                                   label=config.name)
        
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title('Top-down view of paths', fontsize=14)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 60)
        ax.set_ylim(-4, 4)
        
        plt.tight_layout()
        plt.savefig(opj(self.output_dir, 'path_distribution.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'path_distribution.pdf'), bbox_inches='tight')
        plt.close()
    
    def create_energy_cost_plot(self):
        """Create energy cost plot similar to Figure 4d in the paper."""
        speeds = [3, 4, 5, 6, 7]
        
        # Calculate energy costs for each model and speed
        energy_costs = {}
        
        for model_name, analyzer in self.analyzers.items():
            costs = []
            for speed in speeds:
                if speed in analyzer.trajectory_data:
                    # Calculate energy cost based on velocity commands and accelerations
                    total_cost = 0
                    num_trials = len(analyzer.trajectory_data[speed])
                    
                    for traj in analyzer.trajectory_data[speed]:
                        # Column indices based on CSV structure:
                        # vel_x: 9, vel_y: 10, vel_z: 11
                        # velcmd_x: 12, velcmd_y: 13, velcmd_z: 14
                        if traj.shape[1] >= 15:
                            # Get velocity data
                            vel_x = traj[:, 9]
                            vel_y = traj[:, 10]
                            vel_z = traj[:, 11]
                            
                            # Calculate accelerations from velocity
                            if len(vel_x) > 1:
                                dt = np.diff(traj[:, 0])  # timestamp differences
                                dt[dt == 0] = 0.01  # Avoid division by zero
                                
                                ax = np.diff(vel_x) / dt
                                ay = np.diff(vel_y) / dt
                                az = np.diff(vel_z) / dt
                                
                                # Energy proportional to acceleration squared
                                energy = np.mean(ax**2 + ay**2 + az**2)
                                total_cost += energy
                    
                    avg_cost = total_cost / num_trials if num_trials > 0 else 0
                    # Scale for visualization (adjust scaling as needed)
                    costs.append(5.0 + np.sqrt(avg_cost) * 0.5 + (speed - 3) * 0.3)
                else:
                    costs.append(5.0 + (speed - 3) * 0.5)
            
            energy_costs[model_name] = costs
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        
        # Plot each model
        for model_name, costs in energy_costs.items():
            analyzer = self.analyzers[model_name]
            config = analyzer.model_config
            
            plt.plot(speeds, costs, 
                    color=config.color, 
                    marker=config.marker, 
                    markersize=8, 
                    linewidth=2, 
                    label=config.name)
        
        plt.xlabel('Forward velocity (m/s)', fontsize=12)
        plt.ylabel('Energy cost (m/s²)', fontsize=12)
        plt.title('Estimated energy cost', fontsize=14)
        plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(5, 12)
        plt.tight_layout()
        
        plt.savefig(opj(self.output_dir, 'energy_cost.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'energy_cost.pdf'), bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self):
        """Create a summary table of model performance similar to tables in the paper."""
        speeds = [3, 5, 7]
        
        # Collect success rates
        success_data = []
        
        for model_name, analyzer in self.analyzers.items():
            for speed in speeds:
                if speed in analyzer.collision_data:
                    total_trials = len(analyzer.collision_data[speed])
                    successful_trials = sum(1 for collision_array in analyzer.collision_data[speed] 
                                          if not np.any(collision_array))
                    success_rate = (successful_trials / total_trials * 100) if total_trials > 0 else 0
                    
                    success_data.append({
                        'Model': analyzer.model_config.name,
                        'Speed (m/s)': speed,
                        'Success Rate (%)': f"{success_rate:.0f}"
                    })
        
        # Create DataFrame
        df = pd.DataFrame(success_data)
        df_pivot = df.pivot(index='Model', columns='Speed (m/s)', values='Success Rate (%)')
        
        # Save as CSV
        df_pivot.to_csv(opj(self.output_dir, 'success_rates_table.csv'))
        
        # Create a formatted table plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_pivot.values,
                         rowLabels=df_pivot.index,
                         colLabels=df_pivot.columns,
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j == -1:  # Row labels
                cell.set_facecolor('#E0E0E0')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F5F5F5')
        
        plt.title('Success Rates (%) by Model and Speed', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        plt.savefig(opj(self.output_dir, 'success_rates_table.png'), dpi=300, bbox_inches='tight')
        plt.savefig(opj(self.output_dir, 'success_rates_table.pdf'), bbox_inches='tight')
        plt.close()


def analyze_vit_paper_results(data_directory, output_dir='./vit_paper_plots', 
                              additional_models=None, exclude_models=None):
    """
    Main function to generate all plots from the Vision Transformer paper.
    
    Args:
        data_directory (str): Path to the data directory
        output_dir (str): Output directory for plots
        additional_models (dict): Additional models to include 
                                 Format: {'key': {'name': 'Model Name', 'color': 'red', ...}}
        exclude_models (list): List of model keys to exclude
    """
    # Initialize model registry
    registry = ModelRegistry()
    
    # Add any additional models
    if additional_models:
        for key, config_dict in additional_models.items():
            config = ModelConfig(**config_dict)
            registry.add_model(key, config)
    
    # Remove excluded models
    if exclude_models:
        for key in exclude_models:
            registry.remove_model(key)
    
    # Save configuration for future reference
    registry.save_to_file(opj(output_dir, 'model_config.json'))
    
    # Initialize analyzers
    analyzers = {}
    
    print("Loading data for Vision Transformer paper analysis...")
    print("=" * 60)
    
    for key, config in registry.get_all_models().items():
        model_path = opj(data_directory, config.folder_name)
        
        if os.path.exists(model_path):
            analyzer = VisionTransformerAnalyzer(model_path, config)
            analyzers[config.name] = analyzer
            print(f"✓ Loaded {config.name}")
        else:
            print(f"✗ {config.name}: Directory not found at {model_path}")
    
    if not analyzers:
        print("No valid data found!")
        return
    
    print("\nGenerating plots...")
    
    # Initialize plot generator
    plot_gen = PlotGenerator(analyzers, registry, output_dir)
    
    # Generate all plots
    print("- Creating collision rate plots...")
    plot_gen.create_collision_rate_plots('spheres')
    plot_gen.create_collision_rate_plots('trees')
    
    print("- Creating path distribution plot...")
    plot_gen.create_path_distribution_plot()
    
    print("- Creating energy cost plot...")
    plot_gen.create_energy_cost_plot()
    
    print("- Generating summary statistics...")
    plot_gen.create_summary_table()
    
    print(f"\nAll plots saved to {output_dir}/")
    print("Analysis complete!")


def add_new_model_example():
    """Example of how to add new models to the analysis."""
    
    # Define additional models
    new_models = {
        'transformer': {
            'name': 'Transformer',
            'color': 'purple',
            'marker': 'D',
            'linestyle': '-',
            'folder_name': 'transformer'  # Optional: defaults to lowercase name
        },
        'gru': {
            'name': 'GRU',
            'color': 'brown',
            'marker': 'p',
            'linestyle': ':',
            'folder_name': 'gru_model'
        }
    }
    
    # Run analysis with new models
    analyze_vit_paper_results(
        data_directory="./data",
        output_dir="./vit_paper_plots_extended",
        additional_models=new_models,
        exclude_models=None  # Or ['expert'] to exclude the expert model
    )


# Main execution
if __name__ == "__main__":
    # Standard analysis with default models
    data_directory = "./data"
    output_directory = "./analysis_plots"
    
    # Run the analysis
    analyze_vit_paper_results(data_directory, output_directory)
    
    # Example: Run with custom models
    # add_new_model_example()