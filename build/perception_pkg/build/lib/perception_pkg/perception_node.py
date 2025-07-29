import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import struct
import open3d as o3d

# --- 상수 정의 (SCurveConeNavigator에서 가져옴) ---
CONE_MIN_HEIGHT = -0.85  # 라바콘의 최소 높이 (미터) - 라이다 높이 0.8m 고려
CONE_MAX_HEIGHT = -0.05  # 라바콘의 최대 높이 (미터) - 라이다 높이 0.8m 고려
CONE_CLUSTER_DISTANCE_THRESHOLD = 0.6 # 클러스터링 시 같은 라바콘으로 간주할 최대 거리 (미터)
MIN_POINTS_IN_CONE_CLUSTER = 3      # 클러스터를 라바콘으로 간주할 최소 포인트 수
GROUND_SEGMENTATION_THRESHOLD = 0.07 # RANSAC 지면 분할을 위한 거리 임계값 (미터)
OUTLIER_NB_NEIGHBORS = 20            # 이상치 제거를 위한 이웃 포인트 수
OUTLIER_STD_RATIO = 2.0              # 이상치 제거를 위한 표준편차 비율

class PerceptionNode(Node):
    """LiDAR 포인트 클라우드를 처리하여 라바콘을 감지하는 노드"""
    def __init__(self):
        super().__init__('perception_node')
        self.get_logger().info("Perception Node 시작됨")

        # ROS 파라미터 선언
        self.declare_parameter('pointcloud_topic', '/lidar/points')
        pointcloud_topic = self.get_parameter('pointcloud_topic').get_parameter_value().string_value

        # LiDAR 포인트 클라우드 구독자 생성
        self.subscription = self.create_subscription(
            PointCloud2,
            pointcloud_topic,
            self.pointcloud_callback,
            10) # QoS 큐 크기 10
        
        # 감지된 라바콘 발행자 생성
        self.cone_publisher = self.create_publisher(MarkerArray, '/detected_cones', 10)

    def convert_pointcloud2_to_numpy(self, msg):
        """PointCloud2 메시지를 NumPy 배열 (x, y, z)로 변환합니다.
        """
        point_step = msg.point_step # 각 포인트의 바이트 크기
        data = msg.data # 포인트 데이터
        points = []
        # 데이터에서 각 포인트의 x, y, z 값을 추출
        for i in range(0, len(data), point_step):
            # float32 형식으로 x, y, z 값을 언팩 (point_step은 최소 12바이트)
            x, y, z = struct.unpack_from('fff', data, i)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def pointcloud_callback(self, msg):
        """LiDAR 포인트 클라우드 메시지 콜백 함수.
        수신된 포인트 클라우드 데이터를 처리하여 라바콘을 감지하고 발행합니다.
        """
        xyz_data = self.convert_pointcloud2_to_numpy(msg)
        
        if xyz_data.shape[0] == 0:
            self.cone_publisher.publish(MarkerArray()) # 빈 마커 배열 발행
            return

        # NumPy 배열을 Open3D 포인트 클라우드 객체로 변환
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_data)

        # --- PCL 기반 라바콘 감지 --- (SCurveConeNavigator에서 가져옴)

        # 1. 지면 분할 (RANSAC 알고리즘 사용)
        plane_model, inliers = pcd.segment_plane(distance_threshold=GROUND_SEGMENTATION_THRESHOLD, 
                                                 ransac_n=3, 
                                                 num_iterations=1000)
        nonground_pcd = pcd.select_by_index(inliers, invert=True)

        if len(nonground_pcd.points) == 0:
            self.cone_publisher.publish(MarkerArray()) # 빈 마커 배열 발행
            return

        # 2. 이상치 제거 (Statistical Outlier Removal)
        cl, ind = nonground_pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, 
                                                           std_ratio=OUTLIER_STD_RATIO)
        filtered_pcd = nonground_pcd.select_by_index(ind)

        if len(filtered_pcd.points) == 0:
            self.cone_publisher.publish(MarkerArray()) # 빈 마커 배열 발행
            return

        # 3. 라바콘 높이 및 전방 필터링
        filtered_xyz = np.asarray(filtered_pcd.points)
        cone_candidate_points = filtered_xyz[(filtered_xyz[:, 2] > CONE_MIN_HEIGHT) & 
                                             (filtered_xyz[:, 2] < CONE_MAX_HEIGHT) & 
                                             (filtered_xyz[:, 1] > -0.5) & (filtered_xyz[:, 0] > -2.0) & (filtered_xyz[:, 0] < 2.0)]

        if cone_candidate_points.shape[0] == 0:
            self.cone_publisher.publish(MarkerArray()) # 빈 마커 배열 발행
            return

        cone_candidate_pcd = o3d.geometry.PointCloud()
        cone_candidate_pcd.points = o3d.utility.Vector3dVector(cone_candidate_points)

        # 4. 클러스터링 (DBSCAN 알고리즘 사용)
        labels = np.array(cone_candidate_pcd.cluster_dbscan(eps=CONE_CLUSTER_DISTANCE_THRESHOLD, 
                                                            min_points=MIN_POINTS_IN_CONE_CLUSTER, 
                                                            print_progress=False))
        max_label = labels.max()
        
        detected_cones_markers = MarkerArray()
        marker_id = 0

        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) >= MIN_POINTS_IN_CONE_CLUSTER:
                cluster_points = cone_candidate_pcd.select_by_index(list(cluster_indices))
                centroid = cluster_points.get_center()

                # 마커 생성
                marker = Marker()
                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "cones"
                marker.id = marker_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = centroid[0]
                marker.pose.position.y = centroid[1]
                marker.pose.position.z = centroid[2] # 라바콘의 실제 Z 위치
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.3 # 대략적인 라바콘 지름
                marker.scale.y = 0.3
                marker.scale.z = 0.5 # 대략적인 라바콘 높이
                marker.color.a = 1.0 # Alpha
                marker.color.r = 1.0 # Red
                marker.color.g = 0.5 # Orange-ish
                marker.color.b = 0.0

                detected_cones_markers.markers.append(marker)
                marker_id += 1
        
        self.cone_publisher.publish(detected_cones_markers)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()