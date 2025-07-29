import rclpy
from rclpy.node import Node
import numpy as np
import socket
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

class LidarSensorNode(Node):
    def __init__(self):
        super().__init__('lidar_sensor_node')
        self.get_logger().info("LiDAR Sensor Node Started")

        # Parameters
        self.declare_parameter('lidar.local_ip', '192.168.1.201')
        self.declare_parameter('lidar.local_port', 2368)
        self.declare_parameter('lidar.channel', 16)
        self.declare_parameter('lidar.frame_id', 'lidar_link')

        self.local_ip = self.get_parameter('lidar.local_ip').get_parameter_value().string_value
        self.local_port = self.get_parameter('lidar.local_port').get_parameter_value().integer_value
        self.channel = self.get_parameter('lidar.channel').get_parameter_value().integer_value
        self.frame_id = self.get_parameter('lidar.frame_id').get_parameter_value().string_value

        # LiDAR specific configurations
        self.block_size = 1206
        self.max_len = 45
        self.vertical_angle_deg = np.array([[-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]])

        # Network setup
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.local_ip, self.local_port))
        self.get_logger().info(f"Listening for LiDAR data on {self.local_ip}:{self.local_port}")

        # Publisher for PointCloud2
        self.publisher_ = self.create_publisher(PointCloud2, '/lidar/points', 10)
        self.timer = self.create_timer(0.1, self.publish_point_cloud)  # 10 Hz

    def sph2cart(self, R, a):
        x = R * np.cos(np.deg2rad(self.vertical_angle_deg)) * np.sin(np.deg2rad(a))
        y = R * np.cos(np.deg2rad(self.vertical_angle_deg)) * np.cos(np.deg2rad(a))
        z = R * np.sin(np.deg2rad(self.vertical_angle_deg))
        return x.reshape([-1]), y.reshape([-1]), z.reshape([-1])

    def get_lidar_data_as_xyz(self):
        try:
            Buffer = b''
            for _ in range(self.max_len):
                UnitBlock, _ = self.socket.recvfrom(self.block_size)
                Buffer += UnitBlock[:1200]

            Buffer_np = np.frombuffer(Buffer, dtype=np.uint8).reshape([-1, 100])

            Azimuth = np.zeros((24 * self.max_len,))
            Azimuth[0::2] = (
                Buffer_np[:, 2].astype(np.float32)
                + 256 * Buffer_np[:, 3].astype(np.float32)
            )
            Azimuth[1::2] = Azimuth[0::2] + 20

            Distance = (
                Buffer_np[:, 4::3].astype(np.float32)
                + 256 * Buffer_np[:, 5::3].astype(np.float32)
            ) * 2

            Azimuth = Azimuth.reshape([-1, 1]) / 100
            Distance = Distance.reshape([-1, self.channel]) / 1000

            x, y, z = self.sph2cart(Distance, Azimuth)

            xyz = np.concatenate(
                [x.reshape([-1, 1]), y.reshape([-1, 1]), z.reshape([-1, 1])],
                axis=1
            ).astype(np.float32)

            return xyz
        except Exception as e:
            self.get_logger().error(f"Failed to process LiDAR data: {e}")
            return None

    def create_point_cloud_msg(self, xyz_points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        point_cloud_msg = PointCloud2(
            header=header,
            height=1,
            width=len(xyz_points),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,  # 3 (xyz) * 4 bytes
            row_step=12 * len(xyz_points),
            data=xyz_points.tobytes()
        )
        return point_cloud_msg

    def publish_point_cloud(self):
        xyz_points = self.get_lidar_data_as_xyz()
        if xyz_points is not None:
            point_cloud_msg = self.create_point_cloud_msg(xyz_points)
            self.publisher_.publish(point_cloud_msg)
            # self.get_logger().info(f"Published point cloud with {len(xyz_points)} points.")

def main(args=None):
    rclpy.init(args=args)
    node = LidarSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()