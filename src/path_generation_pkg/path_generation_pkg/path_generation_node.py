import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from math import atan2, degrees, radians, sin
from scipy.interpolate import interp1d # 스플라인 보간을 위해 추가

# --- 상수 정의 (SCurveConeNavigator에서 가져옴) ---
WHEELBASE = 1.04             # 로봇의 축거 (앞바퀴와 뒷바퀴 사이 거리) (미터)
TARGET_LOOKAHEAD_DISTANCE = 1.5 # 목표 지점을 로봇 전방에서 찾을 거리 (미터)

class PathGenerationNode(Node):
    """감지된 라바콘을 기반으로 주행 경로를 생성하는 노드"""
    def __init__(self):
        super().__init__('path_generation_node')
        self.get_logger().info("Path Generation Node 시작됨")

        # 감지된 라바콘 구독자 생성
        self.cone_subscription = self.create_subscription(
            MarkerArray,
            '/detected_cones',
            self.cone_callback,
            10)
        
        # 생성된 경로 발행자 생성
        self.path_publisher = self.create_publisher(Path, '/global_path', 10)

        # 파라미터
        self.declare_parameter('path_resolution', 0.1) # 경로 웨이포인트 간격 (미터)
        self.path_resolution = self.get_parameter('path_resolution').get_parameter_value().double_value

    def cone_callback(self, msg: MarkerArray):
        """감지된 라바콘 메시지 콜백 함수. 경로를 생성하여 발행합니다.
        """
        if not msg.markers:
            self.get_logger().warn("감지된 라바콘 없음. 경로 생성 불가.")
            self.path_publisher.publish(Path()) # 빈 경로 발행
            return

        cones = []
        for marker in msg.markers:
            cones.append(np.array([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]))
        
        cones = np.array(cones)

        # 라바콘을 X좌표 기준으로 정렬
        cones_sorted_by_x = cones[cones[:, 0].argsort()]

        # 라바콘을 좌/우로 분류 (Y좌표 기준)
        left_cones = cones_sorted_by_x[cones_sorted_by_x[:, 1] > 0] # Y > 0 이면 왼쪽
        right_cones = cones_sorted_by_x[cones_sorted_by_x[:, 1] <= 0] # Y <= 0 이면 오른쪽

        if len(left_cones) < 2 or len(right_cones) < 2: # 최소 2쌍의 라바콘이 있어야 S자 경로 생성 가능
            self.get_logger().warn("S자 경로 생성을 위한 충분한 좌/우 라바콘 쌍 없음.")
            self.path_publisher.publish(Path()) # 빈 경로 발행
            return

        # 라바콘 쌍 매칭 및 중간 지점 계산
        # 간단한 매칭: X축 기준으로 가장 가까운 라바콘끼리 매칭
        mid_points = []
        for l_cone in left_cones:
            # 가장 가까운 오른쪽 라바콘 찾기
            distances = np.linalg.norm(right_cones[:, :2] - l_cone[:2], axis=1)
            closest_r_cone_idx = np.argmin(distances)
            r_cone = right_cones[closest_r_cone_idx]

            mid_x = (l_cone[0] + r_cone[0]) / 2.0
            mid_y = (l_cone[1] + r_cone[1]) / 2.0
            mid_points.append([mid_x, mid_y])
        
        mid_points = np.array(mid_points)
        # X축 기준으로 다시 정렬 (경로 순서 보장)
        mid_points = mid_points[mid_points[:, 0].argsort()]

        if len(mid_points) < 2:
            self.get_logger().warn("중간 지점 생성 후 유효한 포인트 부족. 경로 생성 불가.")
            self.path_publisher.publish(Path()) # 빈 경로 발행
            return

        # 스플라인 보간을 위한 X, Y 데이터
        x_coords = mid_points[:, 0]
        y_coords = mid_points[:, 1]

        # 경로의 시작점과 끝점 정의
        # 로봇의 현재 위치 (0,0)에서 시작하여 첫 번째 중간 지점까지의 경로를 추가
        # 그리고 마지막 중간 지점에서 약간 더 나아가는 경로를 추가
        # 이 부분은 실제 로봇의 현재 위치를 받아와야 더 정확함
        # 현재는 로봇이 (0,0)에 있다고 가정하고, 첫 라바콘 쌍의 중간 지점부터 시작
        
        # 경로의 총 길이 계산
        path_length = np.sum(np.linalg.norm(mid_points[1:] - mid_points[:-1], axis=1))
        
        # 보간할 새로운 X 좌표 생성
        # 시작점을 0으로, 끝점을 path_length로 설정
        # num_points = int(path_length / self.path_resolution) + 1
        # new_x = np.linspace(0, path_length, num_points)

        # 현재는 X 좌표를 직접 보간
        num_interp_points = int((x_coords[-1] - x_coords[0]) / self.path_resolution) + 1
        new_x_interp = np.linspace(x_coords[0], x_coords[-1], num_interp_points)

        # 스플라인 보간 함수 생성
        f_interp = interp1d(x_coords, y_coords, kind='cubic', fill_value="extrapolate")
        new_y_interp = f_interp(new_x_interp)

        generated_path = Path()
        generated_path.header.frame_id = msg.header.frame_id # 라바콘 마커의 프레임 ID 사용
        generated_path.header.stamp = self.get_clock().now().to_msg()

        for i in range(len(new_x_interp)):
            pose = PoseStamped()
            pose.header = generated_path.header
            pose.pose.position.x = new_x_interp[i]
            pose.pose.position.y = new_y_interp[i]
            pose.pose.position.z = 0.0 # 지면으로 가정
            # 방향 (Yaw) 계산: 다음 웨이포인트와의 각도
            if i < len(new_x_interp) - 1:
                dx = new_x_interp[i+1] - new_x_interp[i]
                dy = new_y_interp[i+1] - new_y_interp[i]
                yaw = atan2(dy, dx)
            else:
                # 마지막 웨이포인트는 이전 웨이포인트의 방향을 따름
                if len(new_x_interp) > 1:
                    dx = new_x_interp[i] - new_x_interp[i-1]
                    dy = new_y_interp[i] - new_y_interp[i-1]
                    yaw = atan2(dy, dx)
                else:
                    yaw = 0.0
            
            q = self.euler_to_quaternion(0, 0, yaw) # 쿼터니언 변환 함수 필요
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            generated_path.poses.append(pose)

        self.path_publisher.publish(generated_path)
        self.get_logger().info(f"경로 생성 완료: {len(generated_path.poses)} 웨이포인트.")

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * sin(pitch/2) * sin(yaw/2)
        qy = math.cos(roll/2) * sin(pitch/2) * math.cos(yaw/2) + sin(roll/2) * math.cos(pitch/2) * sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * sin(yaw/2) - sin(roll/2) * sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + sin(roll/2) * sin(pitch/2) * sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = PathGenerationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
