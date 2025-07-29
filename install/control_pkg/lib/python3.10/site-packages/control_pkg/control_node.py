import rclpy # type: ignore
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, Odometry
import math
import numpy as np # KD-Tree를 위해 numpy 필요
from scipy.spatial import KDTree # KD-Tree 임포트

# find_nearest_waypoint 함수는 이제 클래스 내부에서 KD-Tree를 사용하므로 외부 함수는 필요 없음

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')

        # 파라미터 선언
        self.declare_parameter('stanley_k_base', 0.5)       # Stanley 제어 기본 게인
        self.declare_parameter('stanley_k_speed_factor', 0.1) # Stanley 게인 속도 민감도
        self.declare_parameter('pid_kp', 0.8)          # PID 제어 비례(P) 게인
        self.declare_parameter('pid_ki', 0.1)          # PID 제어 적분(I) 게인
        self.declare_parameter('pid_kd', 0.15)         # PID 제어 미분(D) 게인
        self.declare_parameter('max_speed', 25.0 / 3.6) # 주행 최고 속도 (m/s)
        self.declare_parameter('min_speed', 5.0 / 3.6)  # 주행 최저 속도 (m/s)
        self.declare_parameter('lookahead_distance', 5.0) # 곡률 계산을 위해 내다볼 거리 (m)

        # 파라미터 값 가져오기
        self.k_base = self.get_parameter('stanley_k_base').get_parameter_value().double_value
        self.k_speed_factor = self.get_parameter('stanley_k_speed_factor').get_parameter_value().double_value
        self.Kp = self.get_parameter('pid_kp').get_parameter_value().double_value
        self.Ki = self.get_parameter('pid_ki').get_parameter_value().double_value
        self.Kd = self.get_parameter('pid_kd').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value

        # Subscriber
        self.path_subscriber = self.create_subscription(Path, '/global_path', self.path_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

        # Publisher
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # 변수 초기화
        self.waypoints_x = []
        self.waypoints_y = []
        self.waypoints_yaw = []
        self.kd_tree = None # KD-Tree 객체 초기화
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.pid_integral = 0.0
        self.pid_previous_error = 0.0
        self.last_time = self.get_clock().now()

        self.get_logger().info("Control Node with dynamic speed and KD-Tree has been started.")

    def path_callback(self, msg):
        self.waypoints_x = [pose.pose.position.x for pose in msg.poses]
        self.waypoints_y = [pose.pose.position.y for pose in msg.poses]
        self.waypoints_yaw = []
        for i in range(len(self.waypoints_x) - 1):
            yaw = math.atan2(self.waypoints_y[i+1] - self.waypoints_y[i], self.waypoints_x[i+1] - self.waypoints_x[i])
            self.waypoints_yaw.append(yaw)
        if self.waypoints_yaw:
            self.waypoints_yaw.append(self.waypoints_yaw[-1]) # 마지막 점의 yaw는 이전 점과 동일하게
        
        # KD-Tree 구축
        if self.waypoints_x:
            points = np.array(list(zip(self.waypoints_x, self.waypoints_y)))
            self.kd_tree = KDTree(points)
            self.get_logger().info(f"{len(self.waypoints_x)} waypoints received. KD-Tree built.")
        else:
            self.kd_tree = None
            self.get_logger().warn("Received empty path. KD-Tree not built.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_speed = msg.twist.twist.linear.x

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        if self.waypoints_x and self.kd_tree: # 경로와 KD-Tree가 준비되었을 때만 제어 실행
            self.run_controller()

    def run_controller(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt <= 0:
            return

        # KD-Tree를 사용하여 가장 가까운 웨이포인트 찾기
        if self.kd_tree is None:
            self.get_logger().warn("KD-Tree not available. Cannot run controller.")
            return
        
        # query()는 (거리, 인덱스) 튜플을 반환
        _, nearest_index = self.kd_tree.query([self.current_x, self.current_y])

        # 경로 끝 처리
        if nearest_index >= len(self.waypoints_x) - 1: # 마지막 웨이포인트에 도달했거나 넘어섰을 경우
            self.get_logger().info("Reached end of path.")
            # 경로 끝에 도달하면 정지 명령 발행
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_publisher.publish(cmd_msg)
            return

        # --- 곡률 기반 목표 속도 계산 ---
        max_curvature = 0.0
        # 현재 위치에서 가장 가까운 점부터 lookahead_distance 만큼 앞까지의 곡률을 계산
        start_idx = nearest_index
        # 경로 점 간격을 0.1m로 가정하여 lookahead_distance에 해당하는 점의 개수 계산
        end_idx = min(start_idx + int(self.lookahead_distance / 0.1), len(self.waypoints_x) - 1) 

        for i in range(start_idx, end_idx):
            # 곡률 계산을 위해 최소 3개의 점이 필요
            if i > 0 and i < len(self.waypoints_x) - 1:
                p1_x, p1_y = self.waypoints_x[i-1], self.waypoints_y[i-1]
                p2_x, p2_y = self.waypoints_x[i], self.waypoints_y[i]
                p3_x, p3_y = self.waypoints_x[i+1], self.waypoints_y[i+1]
                
                area = 0.5 * abs(p1_x*(p2_y - p3_y) + p2_x*(p3_y - p1_y) + p3_x*(p1_y - p2_y))
                d1 = math.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
                d2 = math.sqrt((p2_x - p3_x)**2 + (p2_y - p3_y)**2)
                d3 = math.sqrt((p3_x - p1_x)**2 + (p3_y - p1_y)**2)
                
                if d1 * d2 * d3 == 0: # 점들이 일직선상에 있거나 겹치는 경우
                    curvature = 0
                else:
                    curvature = 4 * area / (d1 * d2 * d3)
                
                if curvature > max_curvature: # 가장 큰 곡률 찾기
                    max_curvature = curvature

        # 곡률에 반비례하여 속도 결정
        # 곡률 민감도 조절을 위한 상수 (예: 2.0)는 실제 튜닝 시 조절 필요
        target_speed = self.max_speed - (max_curvature * (self.max_speed - self.min_speed) * 2.0) 
        target_speed = max(self.min_speed, min(target_speed, self.max_speed))
        
        # --- PID 제어 (종방향) ---
        error = target_speed - self.current_speed
        self.pid_integral += error * dt
        derivative = (error - self.pid_previous_error) / dt
        throttle = self.Kp * error + self.Ki * self.pid_integral + self.Kd * derivative
        self.pid_previous_error = error

        # --- Stanley 제어 (횡방향) ---
        # 동적 Stanley 게인 조정 (속도가 낮을수록 k를 높여 경로 추종을 강하게)
        # 0으로 나누는 것을 방지하고, k 값을 합리적인 범위로 제한
        adjusted_k = self.k_base / (self.current_speed + self.k_speed_factor) 
        adjusted_k = max(0.1, min(adjusted_k, 2.0)) # 예시: k는 0.1에서 2.0 사이

        path_yaw = self.waypoints_yaw[nearest_index]

        heading_error = path_yaw - self.current_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        dx = self.waypoints_x[nearest_index] - self.current_x
        dy = self.waypoints_y[nearest_index] - self.current_y
        cross_track_error = math.sqrt(dx**2 + dy**2)
        # 경로의 왼쪽에 있으면 음수, 오른쪽에 있으면 양수 (Stanley 정의에 따름)
        if math.sin(path_yaw - math.atan2(dy, dx)) > 0: 
            cross_track_error = -cross_track_error

        steer_angle = heading_error + math.atan2(adjusted_k * cross_track_error, self.current_speed + 1e-6)

        # --- 제어 명령 발행 ---
        cmd_msg = Twist()
        cmd_msg.linear.x = throttle
        cmd_msg.angular.z = steer_angle
        self.cmd_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
