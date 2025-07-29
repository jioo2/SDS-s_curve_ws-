import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import serial
import time
import struct
import json
import math

class Erp42DriverNode(Node):
    def __init__(self):
        super().__init__('erp42_driver_node')

        # ROS 파라미터 선언
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('max_speed_kmh', 25.0)

        self.port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.max_speed_mps = self.get_parameter('max_speed_kmh').get_parameter_value().double_value / 3.6

        # 차량 제어 상태
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0
        self.seq_num = 0

        # Odometry 관련 변수
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_time = self.get_clock().now()

        # 시리얼 연결 재시도
        self.ser = None
        while rclpy.ok() and self.ser is None:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                self.get_logger().info(f'Connected to {self.port}@{self.baudrate}')
            except serial.SerialException as e:
                self.get_logger().warning(f'Port open failed: {e}, retry in 5s')
                time.sleep(5)
        if not rclpy.ok() or self.ser is None:
            return

        # Subscriber & Publishers
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.response_pub = self.create_publisher(String, '/erp42_response', 10)
        self.status_pub   = self.create_publisher(String, '/erp42_status',   10)
        self.odom_publisher = self.create_publisher(Odometry, '/odom', 10) # Odometry 발행자 추가
        self.create_timer(0.1, self.send_command_and_publish_odom_callback)

        self.get_logger().info("ERP-42 Driver Node started")

    def cmd_vel_callback(self, msg):
        self.target_linear_velocity = msg.linear.x
        self.target_angular_velocity = msg.angular.z

    def parse_status_packet(self, data: bytes):
        if len(data) < 18:
            return None
        b = list(data[:18])

        # 1) A/M 모드
        mode = 'Auto' if b[3]==0x00 else 'Manual'

        # 2) E-Stop
        estop = bool(b[4])

        # 3) Gear
        gear_map = {0:'Neutral',1:'Forward',2:'Reverse'}
        gear = gear_map.get(b[5], b[5])

        # 4) Speed: Speed1 바이트만 사용 (0~200) => 실제 kph = value /10
        speed = b[7] / 10.0  # kph

        # 5) Steer: 두 바이트로 16bit 부호 값, Little-endian으로 해석
        raw_steer = struct.unpack('<h', bytes([b[8], b[9]]))[0]
        steer = raw_steer / 71.0  # deg

        # 6) Brake
        brake = b[10]

        # 7) Encoder: 4바이트 Little-endian 부호 있는 정수
        encoder = struct.unpack('<i', bytes(b[11:15]))[0]

        # 8) Alive
        alive = b[15]

        return {
            'mode': mode,
            'e_stop': estop,
            'gear': gear,
            'speed_kph': speed,
            'steer_deg': steer,
            'brake': brake,
            'encoder': encoder,
            'alive': alive
        }

    def send_command_and_publish_odom_callback(self):
        if not self.ser or not self.ser.is_open:
            return

        # --- Send Command to ERP42 ---
        # gear & speed_cmd
        if self.target_linear_velocity >= 0:
            gear = 0x00
            speed_ratio = self.target_linear_velocity
        else:
            gear = 0x02
            speed_ratio = -self.target_linear_velocity

        if self.max_speed_mps > 0:
            speed_cmd = int(max(0, min(speed_ratio/self.max_speed_mps*200,200)))
        else:
            speed_cmd = 0

        # steer_cmd
        steer_cmd = int(max(-2000, min(-self.target_angular_velocity/0.49*2000, 2000))) # 0.49는 ERP42의 조향비
        steer_val = (steer_cmd + (1<<16)) & 0xFFFF
        steer0, steer1 = steer_val>>8, steer_val&0xFF

        # brake: 최소 1
        brake = 0x01

        packet = [
            0x53,0x54,0x58,  # STX
            0x01,            # A/M
            0x00,            # E-Stop
            gear,            # Gear
            0x00, speed_cmd, # Speed0, Speed1
            steer0, steer1,  # Steer0, Steer1
            brake,           # Brake
            self.seq_num,    # Seq
            0x0D,0x0A        # ETX
        ]
        self.seq_num = (self.seq_num+1)&0xFF

        try:
            buf = bytes(packet)
            self.ser.write(buf)

            # 5) 고정 길이(18바이트)로 읽기
            raw = self.ser.read(18)
            self.response_pub.publish(String(data=str(raw)))

            status = self.parse_status_packet(raw)
            if status is not None:
                self.status_pub.publish(String(data=json.dumps(status)))

                # --- Publish Odometry --- (Using encoder data for position and speed)
                current_time = self.get_clock().now()
                dt = (current_time - self.last_time).nanoseconds / 1e9
                self.last_time = current_time

                # Convert speed from kph to m/s
                linear_speed_mps = status['speed_kph'] / 3.6
                # Convert steer_deg to angular_velocity (approximation for Ackermann)
                # This is a very rough approximation. A proper Ackermann model is needed.
                angular_speed_rads = math.tan(math.radians(status['steer_deg'])) * linear_speed_mps / 1.04 # Assuming wheelbase of 1.04m

                # Update odometry position (simple integration)
                self.odom_x += linear_speed_mps * math.cos(self.odom_yaw) * dt
                self.odom_y += linear_speed_mps * math.sin(self.odom_yaw) * dt
                self.odom_yaw += angular_speed_rads * dt

                odom_msg = Odometry()
                odom_msg.header.stamp = current_time.to_msg()
                odom_msg.header.frame_id = 'odom'
                odom_msg.child_frame_id = 'base_link'

                odom_msg.pose.pose.position.x = self.odom_x
                odom_msg.pose.pose.position.y = self.odom_y
                odom_msg.pose.pose.position.z = 0.0

                q = self.euler_to_quaternion(0, 0, self.odom_yaw)
                odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

                odom_msg.twist.twist.linear.x = linear_speed_mps
                odom_msg.twist.twist.linear.y = 0.0
                odom_msg.twist.twist.linear.z = 0.0
                odom_msg.twist.twist.angular.x = 0.0
                odom_msg.twist.twist.angular.y = 0.0
                odom_msg.twist.twist.angular.z = angular_speed_rads

                self.odom_publisher.publish(odom_msg)

        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            stop = bytes([0x53,0x54,0x58,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x0D,0x0A])
            try: self.ser.write(stop)
            except: pass
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Erp42DriverNode()
    if node.ser:
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
