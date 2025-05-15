#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
import numpy as np
from stable_baselines3 import PPO
import transforms3d.euler
import onnxruntime as ort

def remap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_beta(delta):
    LF = 0.15875
    LR = 0.17145
    return np.arctan(LR * np.tan(delta) / (LF + LR))

def world_to_local(vec, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c,  s],
                    [-s, c]])
    return R @ vec

class ParkingPolicyNode(Node):
    def __init__(self):
        super().__init__('parking_policy_node')

        self.model = self.model = ort.InferenceSession(
            "/home/ubuntu/f1tenth_ws/src/final_6500/models/3waypoint_model.onnx"
        )
        self.get_logger().info("Model loaded successfully")
        self.num_beams = 36
        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        sim = True
        if sim:
            pose_topic = '/ego_racecar/odom'
        else:
            pose_topic = '/pf/pose/odom'

        # Subscriptions
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Odometry, pose_topic, self.odom_callback, qos)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, qos)

        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', qos)

        # State
        self.latest_scan = None
        self.latest_odom = None
        self.last_steering_angle = 0.0

        self.waypoint_pos = None
        self.waypoint_ori = None                     
        self.waypoint_idx = 0                     # Choose 0, 1, or 2

    def odom_callback(self, msg: Odometry):
        if self.waypoint_pos is None or self.latest_scan is None:
            return
        x = msg.pose.position.x
        y = msg.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, yaw = transforms3d.euler.quat2euler([quat.w, quat.x, quat.y, quat.z])
        assert np.abs(yaw) <= np.pi, "Yaws should be in [-pi, pi]"
        pos_error = np.array([x, y] - self.waypoint_pos[self.waypoint_idx])
        body_pos_error = world_to_local(pos_error, yaw)
        pos_error = np.linalg.norm(pos_error, 2)
        ori_error = np.abs(remap_angle(yaw - self.waypoint_ori[self.waypoint_idx]))
        if pos_error < 0.1 and ori_error < np.deg2rad(10.0):
            self.waypoint_idx = min(self.waypoint_idx + 1, 2)
        
        pose_obs = np.array([[body_pos_error[0], body_pos_error[1], yaw]], dtype=np.float32)
        vel_obs = np.array([[msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]], dtype=np.float32)
        heading_obs = np.array([[self.last_steering_angle, get_beta(self.last_steering_angle)]], dtype=np.float32)
        waypoint_obs = np.zeros((1,3), dtype=np.float32)
        waypoint_obs[0, self.waypoint_idx] = 1.0
        
        obs = {
            'scan' : self.latest_scan,
            'pose' : pose_obs,
            'vel' : vel_obs,
            'heading' : heading_obs,
            'waypoint_idx' : waypoint_obs
        }
        self.last_steering_angle, vel = self.run_model(obs)

        drive = AckermannDriveStamped()
        drive.drive.speed = vel
        drive.drive.steering_angle = self.last_steering_angle
        self.drive_pub.publish(drive)
 
    def run_model(self, obs):
        control = self.model.run(None, obs)
        CONTROL_MAX = np.array([0.4189, 2.0])
        control = control[0][0, 0]
        control = control.clip(-1.0, 1.0) * CONTROL_MAX
        steer = float(control[0])
        vel = float(control[1])
        return steer, vel

    def scan_callback(self, msg: LaserScan):
        if self.latest_scan is None:
            self.latest_scan = np.zeros((1, self.num_beams), dtype=np.float32)
            self.get_logger().info("Received first laser scan")
        min_range = msg.range_min
        max_range = msg.range_max
        values = np.array(msg.ranges, dtype=np.float32)
        values = values.clip(min_range, max_range)
        self.latest_scan[0] = values[::len(msg.ranges) // self.num_beams]

    def goal_callback(self, msg: PoseStamped):
        clearance = 0.5
        szx = 0.75
        szy = 0.3
        x = msg.pose.position.x
        y = msg.pose.position.y
        quat = msg.pose.orientation
        _, _, yaw = transforms3d.euler.quat2euler([quat.w, quat.x, quat.y, quat.z])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        T = np.array([[x],[y]])
        if self.waypoint_pos is None:
            self.waypoint_pos = np.zeros((3, 2))
            self.waypoint_ori = np.zeros((3,))
            self.get_logger().info("Received first target pose")

        waypoint_pos = np.array([[clearance + szx / 2, 1.5 * szy],
                                [clearance, 1.5 * szy],
                                [0.0, 0.0]])
        waypoint_pos = R @ waypoint_pos.T + T
        self.waypoint_pos = waypoint_pos.T
        self.waypoint_ori = np.array([yaw, yaw + np.deg2rad(30.0), yaw])
        self.waypoint_ori = remap_angle(self.waypoint_ori)


def main(args=None):
    rclpy.init(args=args)
    node = ParkingPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
