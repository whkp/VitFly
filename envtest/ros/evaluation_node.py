import os
import re
import sys
import yaml
import rospy
import numpy as np

from dodgeros_msgs.msg import Command, QuadState
from envsim_msgs.msg import ObstacleArray
from std_msgs.msg import Empty

from uniplot import plot
import pandas as pd
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, config, exp_name):
        rospy.init_node("evaluator", anonymous=False)
        self.config = config

        self.exp_name = exp_name

        self.xmax = int(self.config["target"])

        self.is_active = False
        self.pos = []
        self.dist = []
        self.time_array = (self.xmax + 1) * [np.nan]

        self.hit_obstacle = False
        self.crash = 0
        self.timeout = self.config["timeout"]
        self.bounding_box = np.reshape(
            np.array(self.config["bounding_box"], dtype=float), (3, 2)
        ).T

        self._initSubscribers(config["topics"])
        self._initPublishers(config["topics"])

        self.ctr = 0
        self.start_time_mark = False

    def _initSubscribers(self, config):
        self.state_sub = rospy.Subscriber(
            "/%s/%s" % (config["quad_name"], config["state"]),
            QuadState,
            self.callbackState,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.obstacle_sub = rospy.Subscriber(
            "/%s/%s" % (config["quad_name"], config["obstacles"]),
            ObstacleArray,
            self.callbackObstacles,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.start_sub = rospy.Subscriber(
            "/%s/%s" % (config["quad_name"], config["start"]),
            Empty,
            self.callbackStart,
            queue_size=1,
            tcp_nodelay=True,
        )

    def _initPublishers(self, config):
        self.finish_pub = rospy.Publisher(
            "/%s/%s" % (config["quad_name"], config["finish"]),
            Empty,
            queue_size=1,
            tcp_nodelay=True,
        )

    def publishFinish(self):
        self.finish_pub.publish()
        self.writeSummary()
        self.printSummary()

    def callbackState(self, msg):

        self.pos_x = msg.pose.position.x

        # mark start time based on position rather than start signal
        if self.pos_x > 0.5 and not self.start_time_mark:
            self.time_array[0] = rospy.get_rostime().to_sec()
            self.start_time_mark = True

        # self.ctr += 1
        # if self.ctr % 30 != 0:
        #     print(f'[evaluator] self.is_active={self.is_active} ; self.time_array[0]={self.time_array[0]:.3f}')

        if not self.is_active:
            return

        pos = np.array(
            [
                msg.header.stamp.to_sec(),
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ]
        )
        self.pos.append(pos)

        bin_x = int(max(min(np.floor(self.pos_x), self.xmax), 0))
        if np.isnan(self.time_array[bin_x]):
            self.time_array[bin_x] = rospy.get_rostime().to_sec()
        if self.pos_x > 60:
            self.is_active = False
            self.publishFinish()

        if rospy.get_time() - self.time_array[0] > self.timeout:
            self.abortRun()

        outside = ((pos[1:] > self.bounding_box[1, :]) | (pos[1:] < self.bounding_box[0, :])
        ).any(axis=-1)
        if (outside == True).any():
            self.abortRun()

    # Note, the start signal may need to be sent multiple times. Sometimes once doesn't work.
    # So, self.time_array[0] is set in callbackState based on position instead.
    def callbackStart(self, msg):
        if not self.is_active:
            self.is_active = True
        # self.time_array[0] = rospy.get_rostime().to_sec()

    def callbackObstacles(self, msg):
        if not self.is_active:
            return

        obs = msg.obstacles[0]
        dist = np.linalg.norm(
            np.array([obs.position.x, obs.position.y, obs.position.z])
        )
        margin = dist - obs.scale
        self.dist.append([msg.header.stamp.to_sec(), margin])
        if margin < 0:
            if not self.hit_obstacle:
                self.crash += 1
                print("Crashed")
            self.hit_obstacle = True
        else:
            self.hit_obstacle = False

    def abortRun(self):
        print("You did not reach the goal!")
        summary = {}
        summary["Success"] = False
        with open("summary.yaml", "w") as f:
            if os.getenv("ROLLOUT_NAME") is not None:
                tmp = {}
                tmp[os.getenv("ROLLOUT_NAME")] = summary
                yaml.safe_dump(tmp, f)
            else:
                yaml.safe_dump(summary, f)
        rospy.signal_shutdown("Completed Evaluation")

    def writeSummary(self):
        """
        - second was logging the whole path.
        """
        return 
        #Time taken throughout the run
        self.timeTaken = self.time_array[-1] - self.time_array[0]

        #Obstacles Collided - just to keep track 
        self.crash = self.crash

        #Distance from nearest obstacles
        dist = np.array(self.dist) #-> time, distance

        #Whole path
        pos = np.array(self.pos) #-> time, x,y,z

        #Since the size of position and nearest obstacle is different, we can't append to same df
        #We also shouldn't interpolate -> can harm the data
        #Saving to two different csv because one's frequency is double than other

        exp_dir = os.path.join("../../labutils/stored_metrics", self.exp_name)
        os.mkdir(exp_dir)

        #XYZ Path File
        # print(pos)
        # print(pos.shape)
        pathFile = os.path.join(exp_dir,"path.csv")
        pd.DataFrame(pos).to_csv(pathFile)

        pathPlots = os.path.join(exp_dir,"XYZ Plots.png")
        _, axs = plt.subplots(3, 1, figsize=(16, 20))
        pos = pos.T
        axs[0].plot(pos[1],pos[2])
        axs[0].set_xlabel("X [m]")
        axs[0].set_ylabel("Y [m]")
        axs[0].set_title("TOP-DOWN; XY")

        axs[1].plot(pos[2], pos[3])
        axs[1].set_xlabel("Y [m]")
        axs[1].set_ylabel("Z [m]")
        axs[1].set_title("HEAD-ON; YZ")
        axs[1].invert_xaxis()
        
        axs[2].plot(pos[1], pos[3])
        axs[2].set_xlabel("X [m]")
        axs[2].set_ylabel("Z [m]")
        axs[2].set_title("SIDE-VIEW; ZX")

        plt.savefig(pathPlots)

        #Distance to Obstacle File
        distFile = os.path.join(exp_dir,"dist.csv")
        nearestDistPlots = os.path.join(exp_dir,"nearestDist.png")
        pd.DataFrame(dist).to_csv(distFile)
        plt.figure()
        plt.plot(dist[:, 0] - self.time_array[0],dist[:, 1])
        plt.xlabel("time (s)");plt.ylabel("Distance from Obstacles [m]")
        plt.savefig(nearestDistPlots)

        # save trainset folder name so more stats can be extracted later
        subdirs = sorted(os.listdir('/home/dhruv/icra22_competition_ws/src/agile_flight/envtest/ros/train_set'))
        stats_dir = subdirs[-1]

        #Time Taken and num collisions Dat File
        Scalarfile = os.path.join(exp_dir,"scalarMetrics.dat")
        with open(Scalarfile, "a") as file:
            file.write(str( float(self.timeTaken) ) + ", " + str(int(self.crash)) + ", " + stats_dir + "\n")


    def printSummary(self):
        
        # 创建以时间和模型名称命名的文件夹来存储结果
        import time
        from datetime import datetime
        import argparse
        
        # 获取模型名称（从命令行参数）
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_type', type=str, default="unknown_model")
        args, unknown = parser.parse_known_args()
        model_type = args.model_type
        
        # 创建结果目录,放到evaluate/results_<timestamp>_<model_type>文件夹下
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "evaluate", f"results_{timestamp}_{model_type}"
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # 计算结果和生成摘要
        ttf = self.time_array[-1] - self.time_array[0]
        summary = {}
        summary["Success"] = True if self.crash == 0 else False
        print("You reached the goal in %5.3f seconds" % ttf)
        summary["time_to_finish"] = ttf
        summary["model_type"] = model_type
        summary["timestamp"] = timestamp
        
        print("Your intermediate times are:")
        print_distance = 10
        summary["segment_times"] = {}
        for i in range(print_distance, self.xmax + 1, print_distance):
            print("    %2i: %5.3fs " % (i, self.time_array[i] - self.time_array[0]))
            summary["segment_times"]["%i" % i] = self.time_array[i] - self.time_array[0]
        
        print("You hit %i obstacles" % self.crash)
        summary["number_crashes"] = self.crash
        
        # 保存摘要到结果目录
        summary_path = os.path.join(results_dir, "summary.yaml")
        with open(summary_path, "w") as f:
            if os.getenv("ROLLOUT_NAME") is not None:
                tmp = {}
                tmp[os.getenv("ROLLOUT_NAME")] = summary
                yaml.safe_dump(tmp, f)
            else:
                yaml.safe_dump(summary, f)
        
        # 同时保留原始位置的摘要
        with open("summary.yaml", "w") as f:
            if os.getenv("ROLLOUT_NAME") is not None:
                tmp = {}
                tmp[os.getenv("ROLLOUT_NAME")] = summary
                yaml.safe_dump(tmp, f)
            else:
                yaml.safe_dump(summary, f)

        if not self.config["plots"]:
            rospy.signal_shutdown("Completed Evaluation")
            return

        print("Here is a plot of your trajectory in the xy plane")
        pos = np.array(self.pos)
        plot(xs=pos[:, 1], ys=pos[:, 2], color=True)
        
        # 保存轨迹图
        if len(pos) > 0:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.plot(pos[:, 1], pos[:, 2])
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title('XY Trajectory')
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, "trajectory_xy.png"))
            
            # 保存3D轨迹图
            if pos.shape[1] > 3:  # 确保有Z坐标
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(pos[:, 1], pos[:, 2], pos[:, 3])
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.set_zlabel('Z [m]')
                ax.set_title('3D Trajectory')
                plt.savefig(os.path.join(results_dir, "trajectory_3d.png"))

        print("Here is a plot of your average velocity per 1m x-segment")
        x = np.arange(1, self.xmax + 1)
        dt = np.array(self.time_array)
        y = 1 / (dt[1:] - dt[0:-1])
        plot(xs=x, ys=y, color=True)
        
        # 保存速度图
        plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        plt.xlabel('Distance [m]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Average Velocity per Segment')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, "velocity.png"))

        print("Here is a plot of the distance to the closest obstacles")
        dist = np.array(self.dist)
        if len(dist) > 0:
            plot(xs=dist[:, 0] - self.time_array[0], ys=dist[:, 1], color=True)
            
            # 保存障碍物距离图
            plt.figure(figsize=(10, 6))
            plt.plot(dist[:, 0] - self.time_array[0], dist[:, 1])
            plt.xlabel('Time [s]')
            plt.ylabel('Distance to Obstacle [m]')
            plt.title('Minimum Distance to Obstacles')
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, "obstacle_distance.png"))
        
        # 保存原始数据供后续分析
        if len(pos) > 0:
            np.savetxt(os.path.join(results_dir, "trajectory.csv"), pos, delimiter=',', 
                       header="time,x,y,z", comments='')
        
        if len(dist) > 0:
            np.savetxt(os.path.join(results_dir, "obstacle_distances.csv"), dist, delimiter=',',
                       header="time,distance", comments='')
            
        # 报告保存位置
        print(f"Results saved to: {results_dir}/")

        rospy.signal_shutdown("Completed Evaluation")


if __name__ == "__main__":
    with open("./evaluation_config.yaml") as f:
        config = yaml.safe_load(f)
    
    # experiment name is passed in as argument in batched rollouts,
    # otherwise it is the current datetime
    from datetime import datetime
    exp_name = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('d%m_%d_t%H_%M')
    Evaluator(config, exp_name)
    rospy.spin()
