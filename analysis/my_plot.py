#绘制xy轴 xz轴以及三维图
import glob, os, sys, time
from os.path import join as opj
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

class gen_plot_data(object):

    def __init__(self, traj_folders, obs_folder):

        self.num_folders = len(traj_folders)
        self.traj_metadata = []
        print(f"Reading {self.num_folders} folders")
        print(f"reading {traj_folders}")

        for folder in traj_folders:
            try:
                # 读取元数据和碰撞数据
                metadata = np.genfromtxt(folder, delimiter=",", dtype=np.float64)[1:, :-1]
                coll_data = np.genfromtxt(folder, delimiter=",", dtype=bool)[1:, -1]
                # 合并元数据和碰撞数据
                self.traj_metadata.append(np.column_stack((metadata, coll_data)))
            except Exception as e:
                print(f"Error reading {folder}: {e}")

        self.traj_metadata = np.row_stack(self.traj_metadata, dtype=np.float64)

        self.obs_trajdata = np.genfromtxt(opj(obs_folder, "static_obstacles.csv"), delimiter=",", dtype=np.float64)[:, 1:]

    def get_collision_count(self,):

        obstacle_count = 0.
        total_obs_duration = 0.
        total_obs_timesteps = 0

        total_obs_timesteps = len(np.where(self.traj_metadata[:, -1] == True)[0])

        start_collision_t = 0.0
        for i in range(len(self.traj_metadata)-1):
            if(self.traj_metadata[i, -1] == False and self.traj_metadata[i+1, -1] == True):
                start_collision_t = self.traj_metadata[i, 1]
            if(self.traj_metadata[i, -1] == True and self.traj_metadata[i+1, -1] == False):
                collision_time = self.traj_metadata[i, 1] - start_collision_t
                #obstacle_count += np.ceil(collision_time)
                obstacle_count += 1
                total_obs_duration += collision_time
                # if(collision_time > 0.3): 
                #     obstacle_count += np.ceil(collision_time)

        #print(f"Average collision count: {obstacle_count / 10}")

        #print(traj_metadata)
                
        ret_data1 = 0 if obstacle_count == 0 else obstacle_count / self.num_folders
        ret_data2 = 0 if obstacle_count == 0 else total_obs_duration / (obstacle_count * self.num_folders)
        ret_data3 = 0 if obstacle_count == 0 else total_obs_timesteps / (obstacle_count * self.num_folders)

        return ret_data1, ret_data2, ret_data3, (obstacle_count, total_obs_duration, total_obs_timesteps, self.num_folders)
    
    
    def get_traj_stats(self,):
        
        x_bins = np.linspace(0, 59, 60)

        x_digitized = np.digitize(self.traj_metadata[:, 7], x_bins)

        traj_stats_data = np.zeros((len(x_bins), 4))

        for i in range(len(x_bins)):

            x_indices = np.where(x_digitized == i)[0]
            if not len(x_indices): continue
            corresponding_yz_data = self.traj_metadata[x_indices, 8:10]

            #print(x_indices)
            #print(corresponding_yz_data)

            traj_stats_data[i, 0:2] = np.mean(corresponding_yz_data, axis=0)
            traj_stats_data[i, 2:] = np.std(corresponding_yz_data, axis=0)

        #print(x_digitized)
            
        traj_stats_data = traj_stats_data[~np.all(traj_stats_data == 0, axis=1)]

        self.traj_stats = traj_stats_data
        
        return traj_stats_data, self.traj_metadata

    def plot_sphere(self, pos, radius, yz_mean, proj=True, alpha=0.5):
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = radius * np.cos(u)*np.sin(v) + pos[0]
        # y = (radius/5) * np.sin(u)*np.sin(v) + pos[1]
        # z = (radius/10) * np.cos(v) + pos[2]
        y = (radius/5) * np.sin(u)*np.sin(v) + pos[1]
        z = (radius/10) * np.cos(v) + pos[2]

        #rgb = ls.shade(x, cmap=cm.Wistia, vert_exag=0.1, blend_mode='soft')
        # blend shade
        #bsl = ls.blend_hsv(rgb, np.expand_dims(x*0.8, 2))

        ax.plot_surface(x, y, z, color='k', alpha=alpha,)
        if proj:
            ax.contourf(x, y, z, zdir="z", offset=0, colors='r', alpha=0.3)
            ax.contourf(x, y, z, zdir="y", offset=5, colors='r', alpha=0.3)

    def plot_2d3d_traj(self ,c1="g", c2="k", with_obs=True):

        self.get_traj_stats()

        for i in range(len(self.traj_stats)-1):

            u = np.linspace(i*1, (i+1)*1, 20)
            t = np.linspace(0, 2 * np.pi, 20)

            t_c, u_c = np.meshgrid(t, u)

            std_dev_y_line = self.traj_stats[i, 2] + (( self.traj_stats[i+1, 2] - self.traj_stats[i, 2] ) / (1)) * (u_c - (i * 1))
            std_dev_z_line = self.traj_stats[i, 3] + (( self.traj_stats[i+1, 3] - self.traj_stats[i, 3] ) / (1)) * (u_c - (i * 1))

            mean_y_line = self.traj_stats[i, 0] + (( self.traj_stats[i+1, 0] - self.traj_stats[i, 0] ) / (1)) * (u_c - (i * 1))
            mean_z_line = self.traj_stats[i, 1] + (( self.traj_stats[i+1, 1] - self.traj_stats[i, 1] ) / (1)) * (u_c - (i * 1))

            #x_c = u_c * std_dev_y_line * (np.cos(t_c))  + mean_y_line
            #y_c = u_c * std_dev_z_line * (np.sin(t_c))  + mean_z_line
            x_c = std_dev_y_line * (np.cos(t_c))  + mean_y_line
            y_c = std_dev_z_line * (np.sin(t_c))  + mean_z_line

            ax.plot_surface(u_c, x_c, y_c, alpha=0.3, color=c1,)
            ax.contourf(u_c, x_c, y_c, zdir="z", offset=0, colors=c1, alpha=0.3)
            ax.contourf(u_c, x_c, y_c, zdir="y", offset=5, colors=c1, alpha=0.3)

            if with_obs and i%4 == 0:
                curr_xyz = np.insert(self.traj_stats[i, 0:2], 0, i)
                #Get the indices of two closest obstacles to the curr_xyz
                closest_obs_indices = np.argpartition(np.linalg.norm(curr_xyz - self.obs_trajdata[:, 0:3], axis=1), 1)[:2]

                for k in range(len(closest_obs_indices)):
                    self.plot_sphere(self.obs_trajdata[closest_obs_indices[k], 0:3], self.obs_trajdata[closest_obs_indices[k], -1], self.traj_stats[i, 0:2])
        
        ax.plot(range(len(self.traj_stats)), self.traj_stats[:, 0], self.traj_stats[:, 1], color=c2)
    
    def plot_2d_xy_traj(self, color="r"):
        self.get_traj_stats()
        x_axis = np.linspace(0, 59, self.traj_stats.shape[0])

        ax.plot(x_axis, self.traj_stats[:, 0], color=color)
        ax.fill_between(x_axis, self.traj_stats[:, 0] - self.traj_stats[:, 2], self.traj_stats[:, 0] + self.traj_stats[:, 2], color=color, alpha=0.3, linestyle="dashed", label="_nolegend_")

    def plot_2d_xz_traj(self, color="r"):
        self.get_traj_stats()
        x_axis = np.linspace(0, 59, self.traj_stats.shape[0])

        ax.plot(x_axis, self.traj_stats[:, 1], color=color)
        ax.fill_between(x_axis, self.traj_stats[:, 1] - self.traj_stats[:, 3], self.traj_stats[:, 1] + self.traj_stats[:, 3], color=color, alpha=0.3, linestyle="dashed", label="_nolegend_")

    def plot_3d_traj(self, color="r", with_obs=True):
        self.get_traj_stats()
        x_axis = np.linspace(0, 59, self.traj_stats.shape[0])
        y_axis = self.traj_stats[:, 0]
        z_axis = self.traj_stats[:, 1]

        ax.plot(x_axis, y_axis, z_axis, color=color)

        for i in range(len(self.traj_stats)-1):
            if with_obs and i%5 == 0:
                    curr_xyz = np.insert(self.traj_stats[i, 0:2], 0, i)
                    #Get the indices of two closest obstacles to the curr_xyz
                    closest_obs_indices = np.argpartition(np.linalg.norm(curr_xyz - self.obs_trajdata[:, 0:3], axis=1), 1)[:2]

                    for k in range(len(closest_obs_indices)):
                        self.plot_sphere(self.obs_trajdata[closest_obs_indices[k], 0:3], self.obs_trajdata[closest_obs_indices[k], -1], self.traj_stats[i, 0:2], proj=False, alpha=0.2)

        

if __name__ == "__main__":
    ele, azim = 32, -48
    vitlstm_folder = "/home/hkp/ws/vitfly_ws/src/vitfly/analysis/data/vit"
    expert_folder = "/home/hkp/ws/vitfly_ws/src/vitfly/analysis/data/expert/"
    light_vit_folder = "/home/hkp/ws/vitfly_ws/src/vitfly/analysis/data/lightvit/"

    obstacle_folder = "/home/hkp/ws/vitfly_ws/src/vitfly/flightmare/flightpy/configs/vision/spheres_medium/environment_0"

    vitlstm_traj_folders = sorted(glob.glob(opj(vitlstm_folder, "*")))
    expert_traj_folders = sorted(glob.glob(opj(expert_folder, "*")))
    light_vit_traj_folders = sorted(glob.glob(opj(light_vit_folder, "*")))


    vit_data = gen_plot_data(vitlstm_traj_folders, obstacle_folder)
    expert_data = gen_plot_data(expert_traj_folders, obstacle_folder)
    lightvit_data = gen_plot_data(light_vit_traj_folders, obstacle_folder)


    fig = plt.figure(num=8, figsize=(35/2, 25/2))

    sns.set_context("talk")

    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)
    plt.rc('axes', labelsize=50)
    plt.rc('axes', titlesize=55)
    plt.rc('legend', fontsize=35)

    fig.tight_layout()
    ax = fig.gca()
    ax.grid(which = "major", linewidth = 1, alpha=1.)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.2)
    ax.minorticks_on()
    
    lightvit_data.plot_2d_xy_traj(color="g")
    vit_data.plot_2d_xy_traj(color="gray")
    expert_data.plot_2d_xy_traj(color="saddlebrown")

    ax.set_xlabel("x-axis (m)", labelpad=-3.0)
    ax.set_ylabel("y-axis (m)")
    ax.set_title("Variation of trajectory on the x-y plane")
    plt.legend(["LightVit","ViT+LSTM",  "Expert"], loc="best", fancybox=True)
    plt.savefig("./plots_modified_md/traj_2d_xy.pdf")
    plt.savefig("./plots_modified_md/traj_2d_xy.png", dpi=900)


    #fig = plt.figure(9)
    fig = plt.figure(num=9, figsize=(35/2, 25/2))

    sns.set_context("talk")

    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)
    plt.rc('axes', labelsize=50)
    plt.rc('axes', titlesize=55)
    plt.rc('legend', fontsize=35)

    fig.tight_layout()
    ax = fig.gca()
    ax.grid(which = "major", linewidth = 1, alpha=1)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.2)
    ax.minorticks_on()

    lightvit_data.plot_2d_xz_traj(color="g")
    vit_data.plot_2d_xz_traj(color="gray")
    expert_data.plot_2d_xz_traj(color="saddlebrown")

    ax.set_xlabel("x-axis (m)", labelpad=-3.0)
    ax.set_ylabel("z-axis (m)")
    ax.set_title("Variation of trajectory on the x-z plane")
    plt.legend(["LightVit","ViT+LSTM",  "Expert"], loc="best", fancybox=True)
    plt.savefig("./plots_modified_md/traj_2d_xz.pdf")
    plt.savefig("./plots_modified_md/traj_2d_xz.png", dpi=900)


    #fig = plt.figure(10)
    fig = plt.figure(num=10, figsize=(35/2, 25/2))

    sns.set_context("talk")

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=25)
    plt.rc('axes', titlesize=35)
    plt.rc('legend', fontsize=24)

    fig.tight_layout()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=ele, azim=azim)

    lightvit_data.plot_3d_traj(color="g")
    vit_data.plot_3d_traj(color="gray")
    expert_data.plot_3d_traj(color="saddlebrown")

    ax.set_xlim([0, 60])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel("x-axis (m)", labelpad=10.0)
    ax.set_ylabel("y-axis (m)", labelpad=10.0)
    ax.set_zlabel("z-axis (m)", labelpad=10.0)
    ax.set_title("Trajectories with obstacles")
    plt.legend(["LightVit","ViT+LSTM",  "Expert"], loc="best", fancybox=True)
    plt.savefig("./plots_modified_md/traj_3d.pdf")
    plt.savefig("./plots_modified_md/traj_3d.png", dpi=900)