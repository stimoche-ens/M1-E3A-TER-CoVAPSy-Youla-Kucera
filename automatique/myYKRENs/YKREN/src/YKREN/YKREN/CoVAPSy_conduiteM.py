
from matplotlib.pylab import angle
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from rclpy.node import Node

class NoeudconduiteManette(Node):
    def __init__(self):
        super().__init__('CoVAPSy_conduiteM')

        # ROS interface
        self.__ackermann_publisher = self.create_publisher(AckermannDrive, 'cmd_ackermann', 1)
        self.scan_publisher = self.create_publisher(LaserScan, 'scan_fixe', 1)
        self.create_subscription(Joy, 'joy', self.__on_joystick_acquisition, 1)
        self.create_subscription(LaserScan, 'scan', self.lidar_stable_publication,1)
        self.get_logger().info('noeud cree')

    def __on_joystick_acquisition(self, joy_message):
        self.get_logger().info(f'vit {joy_message.axes[1]:.2f} et dir {joy_message.axes[3]:.2f}')
        command_message = AckermannDrive()
        command_message.speed = joy_message.axes[1]*2 # speed between -2 and 2 m/s, convert from joystick input which is between -1 and 1
        command_message.steering_angle = joy_message.axes[3]*18 # angle between -18 and 18 degrees, convert from joystick input which is between -1 and 1
        if command_message.steering_angle > 18.0:
            command_message.steering_angle = 18.0
        if command_message.steering_angle < -18.0:
            command_message.steering_angle = -18.0
        self.__ackermann_publisher.publish(command_message) # angle in degrees, speed in m/s
        self.get_logger().info(f'v = {command_message.speed:.2f} m/s, dir = {command_message.steering_angle:.2f} deg')

    def angle_to_index(self, angle_deg, scan_msg):
        # function converting angle in degrees to index 
        if angle_deg>180: #modulo 2pi to keep the angle between -180 and 180
            angle_deg+=360
        elif angle_deg<-180: #modulo 2pi to keep the angle between -180 and 180
            angle_deg-=360
        angle_rad = np.deg2rad(angle_deg)
        idx = int(angle_rad / scan_msg.angle_increment) # angles positive à gauche, angles négatifs à droite, angle 0° au milieu
        return idx
    
    def lidar_stable_publication(self, message):
        # function publishing a fixed number of LiDAR points (ex: 201 points from -100° to 100° with a step of 1°)
        LidarTable = list(message.ranges) 

        def generate_median_scan(start_angle=-100, end_angle=100, step=1):
            # function taking the median of the points in each angle range
            angles = range(start_angle, end_angle + step, step)  # inclut end_angle
            median_scan = []
            for i, angle in enumerate(angles): # go over each angle in the range we're interested in
                next_angle = angle + step
                start_idx = self.angle_to_index(angle,message)
                end_idx = self.angle_to_index(next_angle,message)
                self.get_logger().info(f'angle = {angle:.2f}°, start_idx = {start_idx}, end_idx = {end_idx}')

                tab=[]
                for j in range(min(start_idx, end_idx), max(start_idx, end_idx) + 1): # go over the indices corresponding to the angle range
                    tab.append(LidarTable[j])
                if tab: # if there are points in the range, calculate the median, otherwise append NaN
                    median_scan.append(np.median(tab))
                else:
                    median_scan.append(float('nan'))
            return median_scan

        
        # Generate the fixed LiDAR table 
        LidarTable_fixed = generate_median_scan(-100, 100, 1) # 201 points from -100° to 100° with a step of 1°

        # Publication in the Lidar topic
        scan_message = LaserScan()
        scan_message.ranges = LidarTable_fixed
        self.scan_publisher.publish(scan_message)

        # Debug : affichage de -100° et 100°
        idx_minus100 = 0  # -100° est le premier point
        idx_plus100 = len(LidarTable_fixed) - 1  # 100° est le dernier point
        self.get_logger().info(f'-100° : {LidarTable_fixed[idx_minus100]:.2f}, 100° : {LidarTable_fixed[idx_plus100]:.2f}')

def main(args=None):
    rclpy.init(args=args)
    noeud = NoeudconduiteManette()
    rclpy.spin(noeud)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
