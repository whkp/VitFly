import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from os.path import join as opj
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D

class TrajectoryAnalyzer:
    """
    A class to analyze and visualize trajectory data from different ML models.
    
    This class handles:
    - Loading trajectory data from CSV files in subfolders
    - Computing collision statistics
    - Calculating trajectory statistics (mean, std deviation)
    - Creating 2D and 3D visualizations with static obstacles
    """
    
    def __init__(self, data_folder, model_name, obs_folder=None):
        """
        Initialize the analyzer with data from a specific model.
        
        Args:
            data_folder (str): Path to the folder containing subfolders with data.csv files
            model_name (str): Name of the model for identification
            obs_folder (str): Path to folder containing static_obstacles.csv (optional)
        """
        self.model_name = model_name
        self.data_folder = data_folder
        self.obs_folder = obs_folder
        
        # Try to load the trajectory data from subfolders
        try:
            # Get all subfolders containing data.csv files
            subfolder_paths = sorted(glob.glob(opj(data_folder, "*")))
            valid_subfolders = []
            
            for subfolder in subfolder_paths:
                csv_path = opj(subfolder, "data.csv")
                if os.path.exists(csv_path):
                    valid_subfolders.append(subfolder)
            
            if not valid_subfolders:
                print(f"No valid subfolders with data.csv found in {data_folder}")
                self.trajectory_data = None
                return
            
            print(f"Loading data for {model_name} from {len(valid_subfolders)} subfolders")
            
            # Load and combine data from all subfolders
            traj_metadata = []
            
            for folder in valid_subfolders:
                csv_path = opj(folder, "data.csv")
                try:
                    # Read metadata (all columns except last)
                    metadata = np.genfromtxt(csv_path, delimiter=",", dtype=np.float64)[1:, :-1]
                    # Read collision data (last column)
                    coll_data = np.genfromtxt(csv_path, delimiter=",", dtype=bool)[1:, -1]
                    # Combine metadata and collision data
                    combined_data = np.column_stack((metadata, coll_data))
                    traj_metadata.append(combined_data)
                except Exception as e:
                    print(f"Failed to read {csv_path}: {e}")
                    continue
            
            if not traj_metadata:
                raise ValueError("No valid trajectory data found in any subfolder")
            
            # Stack all trajectory data
            self.trajectory_data = np.row_stack(traj_metadata)
            print(f"Successfully loaded {len(self.trajectory_data)} data points for {model_name}")
            print(f"Data shape: {self.trajectory_data.shape}")
            
        except Exception as e:
            print(f"Error loading data for {model_name}: {e}")
            self.trajectory_data = None
            return
        
        # Load obstacle data if provided
        self.obs_data = None
        if obs_folder and os.path.exists(obs_folder):
            try:
                obs_csv_path = opj(obs_folder, "static_obstacles.csv")
                if os.path.exists(obs_csv_path):
                    self.obs_data = np.genfromtxt(obs_csv_path, delimiter=",", dtype=np.float64)[:, 1:]
                    print(f"Loaded {len(self.obs_data)} static obstacles")
                else:
                    print(f"No static_obstacles.csv found in {obs_folder}")
            except Exception as e:
                print(f"Error loading obstacle data: {e}")
                
        # Initialize statistics storage
        self.traj_stats = None
        
    def is_valid(self):
        """Check if the analyzer has valid data."""
        return self.trajectory_data is not None
    
    def analyze_collisions(self):
        """
        Analyze collision patterns in the trajectory data.
        
        Returns:
            dict: Dictionary containing collision statistics
        """
        if not self.is_valid():
            return {"error": "No valid data"}
        
        # Assume last column contains collision information (boolean)
        collision_column = -1
        
        try:
            # Convert collision data to boolean if needed
            collision_data = self.trajectory_data[:, collision_column].astype(bool)
            
            # Count total collision timesteps
            total_collision_timesteps = np.sum(collision_data)
            
            # Count collision events (sequences of consecutive collisions)
            collision_events = 0
            in_collision = False
            collision_durations = []
            start_time = 0
            
            # Assuming time is in column 1 (adjust based on your data structure)
            time_column = 1 if self.trajectory_data.shape[1] > 1 else 0
            
            for i in range(len(collision_data)):
                if collision_data[i] and not in_collision:
                    # Start of a new collision
                    in_collision = True
                    collision_events += 1
                    start_time = self.trajectory_data[i, time_column] if time_column < self.trajectory_data.shape[1] else i
                elif not collision_data[i] and in_collision:
                    # End of collision
                    in_collision = False
                    end_time = self.trajectory_data[i-1, time_column] if time_column < self.trajectory_data.shape[1] else i-1
                    collision_durations.append(end_time - start_time)
            
            # Calculate statistics
            collision_rate = total_collision_timesteps / len(collision_data) if len(collision_data) > 0 else 0
            avg_collision_duration = np.mean(collision_durations) if collision_durations else 0
            
            return {
                "total_timesteps": len(collision_data),
                "collision_timesteps": total_collision_timesteps,
                "collision_events": collision_events,
                "collision_rate": collision_rate,
                "average_collision_duration": avg_collision_duration,
                "collision_durations": collision_durations
            }
            
        except Exception as e:
            print(f"Error analyzing collisions for {self.model_name}: {e}")
            return {"error": str(e)}
    
    def compute_trajectory_statistics(self, x_bins=60):
        """
        Compute trajectory statistics by binning data along x-axis.
        
        Args:
            x_bins (int): Number of bins for x-axis discretization
            
        Returns:
            numpy.ndarray: Array of statistics [mean_y, mean_z, std_y, std_z] for each bin
        """
        if not self.is_valid():
            return None
            
        try:
            # Assume columns are organized as: [index, time, ..., x, y, z, ...]
            # You may need to adjust these column indices based on your actual data structure
            if self.trajectory_data.shape[1] >= 10:
                x_col, y_col, z_col = 7, 8, 9  # Original column indices
            elif self.trajectory_data.shape[1] >= 5:
                x_col, y_col, z_col = 2, 3, 4  # Adjusted for smaller datasets
            else:
                print(f"Warning: Not enough columns in data for {self.model_name}")
                return None
            
            # Create bins for x-axis - using linspace similar to gen_plots.py
            x_data = self.trajectory_data[:, x_col]
            x_bins_edges = np.linspace(0, 59, x_bins)
            
            # Digitize x-coordinates to find which bin each point belongs to
            x_digitized = np.digitize(x_data, x_bins_edges)
            
            # Initialize statistics array
            traj_stats = np.zeros((len(x_bins_edges), 4))  # [mean_y, mean_z, std_y, std_z]
            
            # Calculate statistics for each bin
            for i in range(len(x_bins_edges)):
                bin_indices = np.where(x_digitized == i)[0]
                
                if len(bin_indices) > 0:
                    y_data = self.trajectory_data[bin_indices, y_col]
                    z_data = self.trajectory_data[bin_indices, z_col]
                    
                    traj_stats[i, 0] = np.mean(y_data)  # mean_y
                    traj_stats[i, 1] = np.mean(z_data)  # mean_z
                    traj_stats[i, 2] = np.std(y_data)   # std_y
                    traj_stats[i, 3] = np.std(z_data)   # std_z
            
            # Remove bins with no data (all zeros)
            valid_bins = ~np.all(traj_stats == 0, axis=1)
            traj_stats = traj_stats[valid_bins]
            
            self.traj_stats = traj_stats
            return traj_stats
            
        except Exception as e:
            print(f"Error computing trajectory statistics for {self.model_name}: {e}")
            return None
    
    def plot_sphere(self, ax, pos, radius, alpha=0.2):
        """
        Plot a sphere representing a static obstacle.
        
        Args:
            ax: Matplotlib 3D axis object
            pos: Position [x, y, z] of the sphere center
            radius: Radius of the sphere
            alpha: Transparency of the sphere
        """
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = radius * np.cos(u) * np.sin(v) + pos[0]
        y = (radius/5) * np.sin(u) * np.sin(v) + pos[1]
        z = (radius/10) * np.cos(v) + pos[2]
        
        ax.plot_surface(x, y, z, color='k', alpha=alpha)
    
    def plot_2d_trajectory(self, ax, plane='xy', color='blue', alpha=0.7):
        """
        Plot 2D trajectory projection.
        
        Args:
            ax: Matplotlib axis object
            plane (str): Either 'xy' or 'xz' for the projection plane
            color (str): Color for the trajectory line
            alpha (float): Transparency for the confidence band
        """
        if self.traj_stats is None:
            self.compute_trajectory_statistics()
        
        if self.traj_stats is None:
            print(f"Cannot plot 2D trajectory for {self.model_name}: No valid statistics")
            return
        
        # Create x-axis based on trajectory length
        x_axis = np.linspace(0, 59, len(self.traj_stats))
        
        if plane == 'xy':
            mean_data = self.traj_stats[:, 0]  # mean_y
            std_data = self.traj_stats[:, 2]   # std_y
            ylabel = 'y-axis (m)'
        else:  # 'xz'
            mean_data = self.traj_stats[:, 1]  # mean_z
            std_data = self.traj_stats[:, 3]   # std_z
            ylabel = 'z-axis (m)'
        
        # Plot mean trajectory
        ax.plot(x_axis, mean_data, color=color, label=self.model_name, linewidth=2)
        
        # Plot confidence band (mean ± standard deviation)
        ax.fill_between(x_axis, 
                       mean_data - std_data, 
                       mean_data + std_data, 
                       color=color, alpha=alpha)
        
        ax.set_xlabel('x-axis (m)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    def plot_3d_trajectory(self, ax, color='blue', with_obstacles=True):
        """
        Plot 3D trajectory with optional static obstacles.
        
        Args:
            ax: Matplotlib 3D axis object
            color (str): Color for the trajectory line
            with_obstacles (bool): Whether to plot static obstacles
        """
        if self.traj_stats is None:
            self.compute_trajectory_statistics()
        
        if self.traj_stats is None:
            print(f"Cannot plot 3D trajectory for {self.model_name}: No valid statistics")
            return
        
        # Create coordinate arrays
        x_axis = np.linspace(0, 59, len(self.traj_stats))
        y_axis = self.traj_stats[:, 0]  # mean_y
        z_axis = self.traj_stats[:, 1]  # mean_z
        
        # Plot 3D trajectory
        ax.plot(x_axis, y_axis, z_axis, color=color, label=self.model_name, linewidth=2)
        
        # Plot obstacles if available and requested
        if with_obstacles and self.obs_data is not None:
            for i in range(len(self.traj_stats)):
                if i % 5 == 0:  # Plot obstacles every 5 points to avoid clutter
                    curr_xyz = np.insert(self.traj_stats[i, 0:2], 0, i)
                    
                    # Get the indices of two closest obstacles to the current position
                    distances = np.linalg.norm(curr_xyz - self.obs_data[:, 0:3], axis=1)
                    closest_obs_indices = np.argpartition(distances, min(2, len(self.obs_data)-1))[:2]
                    
                    for k in range(len(closest_obs_indices)):
                        obs_pos = self.obs_data[closest_obs_indices[k], 0:3]
                        obs_radius = self.obs_data[closest_obs_indices[k], -1]
                        self.plot_sphere(ax, obs_pos, obs_radius, alpha=0.2)


def analyze_all_models(data_directory, obs_folder=None):
    """
    Analyze all models in the data directory and create comprehensive visualizations.
    
    Args:
        data_directory (str): Path to the data directory containing model folders
        obs_folder (str): Path to the folder containing static_obstacles.csv (optional)
    """
    
    # Define model configurations
    model_configs = {
        'expert': {'color': 'saddlebrown', 'name': 'Expert'},
        'vit': {'color': 'gray', 'name': 'ViT'},
        'lstm': {'color': 'red', 'name': 'LSTM'},
        'vitlstm': {'color': 'green', 'name': 'ViT+LSTM'},
        'convnet': {'color': 'steelblue', 'name': 'ConvNet'}
    }
    
    # Initialize analyzers for each model
    analyzers = {}
    collision_stats = {}
    
    print("Loading and analyzing trajectory data for all models...")
    
    for model_folder, config in model_configs.items():
        model_path = opj(data_directory, model_folder)
        
        if os.path.exists(model_path):
            analyzer = TrajectoryAnalyzer(model_path, config['name'], obs_folder)
            if analyzer.is_valid():
                analyzers[model_folder] = analyzer
                # Analyze collisions
                collision_stats[model_folder] = analyzer.analyze_collisions()
                print(f"✓ {config['name']}: Loaded successfully")
            else:
                print(f"✗ {config['name']}: Failed to load data")
        else:
            print(f"✗ {config['name']}: Directory not found at {model_path}")
    
    if not analyzers:
        print("No valid data found for any model!")
        return
    
    # Create output directory for plots
    output_dir = "./analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 100})
    
    # 1. Collision Analysis Bar Chart
    create_collision_analysis_plot(collision_stats, model_configs, output_dir)
    
    # 2. 2D Trajectory Plots (XY and XZ planes)
    create_2d_trajectory_plots(analyzers, model_configs, output_dir)
    
    # 3. 3D Trajectory Plot with obstacles
    create_3d_trajectory_plot(analyzers, model_configs, output_dir)
    
    # 4. Statistical Summary
    create_statistical_summary(collision_stats, analyzers, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")


def create_collision_analysis_plot(collision_stats, model_configs, output_dir):
    """Create a comprehensive collision analysis visualization."""
    
    # Extract collision metrics
    models = []
    collision_rates = []
    collision_events = []
    avg_durations = []
    
    for model_folder, stats in collision_stats.items():
        if 'error' not in stats:
            models.append(model_configs[model_folder]['name'])
            collision_rates.append(stats['collision_rate'] * 100)  # Convert to percentage
            collision_events.append(stats['collision_events'])
            avg_durations.append(stats['average_collision_duration'])
    
    if not models:
        print("No valid collision data found")
        return
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Collision Analysis Across Models', fontsize=16, fontweight='bold')
    
    # Plot 1: Collision Rate
    colors = [model_configs[k]['color'] for k in collision_stats.keys() if 'error' not in collision_stats[k]]
    bars1 = ax1.bar(models, collision_rates, color=colors, alpha=0.7)
    ax1.set_ylabel('Collision Rate (%)')
    ax1.set_title('Collision Rate by Model')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, collision_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Number of Collision Events
    bars2 = ax2.bar(models, collision_events, color=colors, alpha=0.7)
    ax2.set_ylabel('Number of Collision Events')
    ax2.set_title('Total Collision Events by Model')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, collision_events):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{int(value)}', ha='center', va='bottom')
    
    # Plot 3: Average Collision Duration
    bars3 = ax3.bar(models, avg_durations, color=colors, alpha=0.7)
    ax3.set_ylabel('Average Duration (time units)')
    ax3.set_title('Average Collision Duration by Model')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, avg_durations):
        if value > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom')
    
    # Plot 4: Safety Ranking (inverse of collision rate)
    safety_scores = [100 - rate for rate in collision_rates]
    bars4 = ax4.bar(models, safety_scores, color=colors, alpha=0.7)
    ax4.set_ylabel('Safety Score (%)')
    ax4.set_title('Safety Ranking (100% - Collision Rate)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars4, safety_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(opj(output_dir, 'collision_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(opj(output_dir, 'collision_analysis.pdf'), bbox_inches='tight')
    plt.close()


def create_2d_trajectory_plots(analyzers, model_configs, output_dir):
    """Create 2D trajectory projection plots."""
    
    # Create XY plane plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_folder, analyzer in analyzers.items():
        color = model_configs[model_folder]['color']
        analyzer.plot_2d_trajectory(ax, plane='xy', color=color)
    
    ax.set_title('Trajectory Variation in XY Plane', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(opj(output_dir, 'trajectory_2d_xy.png'), dpi=300, bbox_inches='tight')
    plt.savefig(opj(output_dir, 'trajectory_2d_xy.pdf'), bbox_inches='tight')
    plt.close()
    
    # Create XZ plane plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_folder, analyzer in analyzers.items():
        color = model_configs[model_folder]['color']
        analyzer.plot_2d_trajectory(ax, plane='xz', color=color)
    
    ax.set_title('Trajectory Variation in XZ Plane', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(opj(output_dir, 'trajectory_2d_xz.png'), dpi=300, bbox_inches='tight')
    plt.savefig(opj(output_dir, 'trajectory_2d_xz.pdf'), bbox_inches='tight')
    plt.close()


def create_3d_trajectory_plot(analyzers, model_configs, output_dir):
    """Create 3D trajectory visualization with obstacles."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for model_folder, analyzer in analyzers.items():
        color = model_configs[model_folder]['color']
        analyzer.plot_3d_trajectory(ax, color=color, with_obstacles=True)
    
    ax.set_xlabel('x-axis (m)')
    ax.set_ylabel('y-axis (m)')
    ax.set_zlabel('z-axis (m)')
    ax.set_title('3D Trajectory Comparison with Static Obstacles', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=-60)
    
    # Set axis limits similar to gen_plots.py
    ax.set_xlim([0, 60])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 5])
    
    plt.tight_layout()
    plt.savefig(opj(output_dir, 'trajectory_3d.png'), dpi=300, bbox_inches='tight')
    plt.savefig(opj(output_dir, 'trajectory_3d.pdf'), bbox_inches='tight')
    plt.close()


def create_statistical_summary(collision_stats, analyzers, output_dir):
    """Create a comprehensive statistical summary."""
    
    summary_data = []
    
    for model_folder, stats in collision_stats.items():
        if 'error' not in stats and model_folder in analyzers:
            analyzer = analyzers[model_folder]
            
            # Compute additional trajectory statistics
            if analyzer.traj_stats is not None:
                avg_y_variation = np.mean(analyzer.traj_stats[:, 2])  # avg std_y
                avg_z_variation = np.mean(analyzer.traj_stats[:, 3])  # avg std_z
                trajectory_smoothness = 1 / (1 + avg_y_variation + avg_z_variation)  # Simple smoothness metric
            else:
                avg_y_variation = 0
                avg_z_variation = 0
                trajectory_smoothness = 0
            
            summary_data.append({
                'Model': analyzer.model_name,
                'Collision Rate (%)': stats['collision_rate'] * 100,
                'Collision Events': stats['collision_events'],
                'Avg Collision Duration': stats['average_collision_duration'],
                'Avg Y Variation': avg_y_variation,
                'Avg Z Variation': avg_z_variation,
                'Trajectory Smoothness': trajectory_smoothness,
                'Safety Score': 100 - (stats['collision_rate'] * 100)
            })
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(summary_data)
    df = df.round(3)
    df.to_csv(opj(output_dir, 'statistical_summary.csv'), index=False)
    
    # Create a summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a heatmap of normalized metrics
    metrics = ['Collision Rate (%)', 'Collision Events', 'Avg Y Variation', 'Avg Z Variation']
    plot_data = df[['Model'] + metrics].set_index('Model')
    
    # Normalize data for better visualization (0-1 scale)
    plot_data_norm = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
    
    im = ax.imshow(plot_data_norm.T, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(plot_data.index)))
    ax.set_xticklabels(plot_data.index, rotation=45)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(plot_data.index)):
            text = ax.text(j, i, f'{plot_data.iloc[j, i]:.2f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Metrics Heatmap (Normalized)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value (0=best, 1=worst for negative metrics)')
    
    plt.tight_layout()
    plt.savefig(opj(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(opj(output_dir, 'metrics_heatmap.pdf'), bbox_inches='tight')
    plt.close()
    
    print("\nStatistical Summary:")
    print(df.to_string(index=False))


# Main execution
if __name__ == "__main__":
    # Set the data directory path
    data_directory = "./data"
    
    # Set the obstacle folder path (optional)
    obstacle_folder = "/home/hkp/ws/vitfly_ws/src/vitfly/flightmare/flightpy/configs/vision/spheres_medium/environment_50"  # Update this path as needed
    
    print("Starting comprehensive trajectory analysis...")
    print("=" * 50)
    
    # Analyze all models with obstacle data
    analyze_all_models(data_directory, obstacle_folder)
    
    print("\nAnalysis complete! Check the 'analysis_plots' directory for visualizations.")