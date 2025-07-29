import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
import sys

from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D # For 2D detections

# YOLOv5 저장소 경로를 PYTHONPATH에 추가
# 이 경로는 s_curve_ws/src/yolov5_repo 로 클론된 YOLOv5 저장소의 루트여야 합니다.
# sys.path.append('/root/s_curve_ws/src/yolov5_repo')

# YOLOv5 모델 로드 (로컬에서 직접 로드)
# from models.common import DetectMultiBackend # 이 라인은 YOLOv5 저장소의 models/common.py에 의존
# from utils.general import non_max_suppression, scale_boxes, xyxy2xywh # 이 라인들은 YOLOv5 저장소의 utils/general.py에 의존

class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')
        self.get_logger().info("Cone Detector Node 시작됨 (YOLOv5)")

        # 파라미터 선언
        self.declare_parameter('model_path', 'models/best.pt')
        self.declare_parameter('image_topic_left', '/camera/left/image_raw')
        self.declare_parameter('image_topic_right', '/camera/right/image_raw')
        self.declare_parameter('detection_topic_left', '/camera/left/cones_2d')
        self.declare_parameter('detection_topic_right', '/camera/right/cones_2d')
        self.declare_parameter('confidence_threshold', 0.5)

        model_path_relative = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic_left = self.get_parameter('image_topic_left').get_parameter_value().string_value
        self.image_topic_right = self.get_parameter('image_topic_right').get_parameter_value().string_value
        self.detection_topic_left = self.get_parameter('detection_topic_left').get_parameter_value().string_value
        self.detection_topic_right = self.get_parameter('detection_topic_right').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # 모델 경로 설정 (패키지 내의 models 디렉토리)
        package_share_directory = os.path.join(os.path.dirname(__file__), '..', '..')
        self.model_full_path = os.path.join(package_share_directory, model_path_relative)

        # YOLOv5 모델 로드 (로컬에서 직접 로드)
        try:
            # YOLOv5 저장소의 models/common.py와 utils/general.py를 직접 사용
            # sys.path에 YOLOv5 저장소 루트를 추가해야 함
            yolov5_repo_path = '/root/s_curve_ws/src/yolov5_repo' # YOLOv5 저장소 클론 경로
            if yolov5_repo_path not in sys.path:
                sys.path.append(yolov5_repo_path)
            
            from models.common import DetectMultiBackend
            from utils.general import non_max_suppression, scale_boxes, xyxy2xywh

            self.model = DetectMultiBackend(self.model_full_path, device=torch.device('cpu')) # CPU 사용
            self.model.eval() # 모델을 평가 모드로 설정
            self.get_logger().info(f"YOLOv5 model loaded from {self.model_full_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv5 model: {e}")
            rclpy.shutdown()
            return

        self.bridge = CvBridge()

        # 구독자 및 발행자
        self.subscription_left = self.create_subscription(
            Image,
            self.image_topic_left,
            lambda msg: self.image_callback(msg, 'left'),
            10)
        self.subscription_right = self.create_subscription(
            Image,
            self.image_topic_right,
            lambda msg: self.image_callback(msg, 'right'),
            10)

        self.publisher_left = self.create_publisher(Detection2DArray, self.detection_topic_left, 10)
        self.publisher_right = self.create_publisher(Detection2DArray, self.detection_topic_right, 10)
        self.detection_image_publisher_left = self.create_publisher(Image, '/camera/left/image_detections', 10)
        self.detection_image_publisher_right = self.create_publisher(Image, '/camera/right/image_detections', 10)

    def image_callback(self, msg, camera_side):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # 이미지 전처리 (YOLOv5 입력 형식에 맞게)
        img = cv2.resize(cv_image, (640, 640)) # YOLOv5 기본 입력 크기
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.model.device) # NumPy to Torch tensor
        img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # Add batch dimension
            img = img.unsqueeze(0)

        # YOLOv5 추론
        pred = self.model(img) # 모델 추론

        # NMS (Non-Maximum Suppression)
        pred = non_max_suppression(pred, self.confidence_threshold, 0.45) # conf_thres, iou_thres

        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        # 결과 파싱 및 이미지에 그리기
        display_image = cv_image.copy() # 원본 이미지 복사하여 그리기

        for det in pred: # per image
            if det is not None and len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], cv_image.shape).round()

                for *xyxy, conf, cls in det:
                    detection = Detection2D()
                    detection.header = msg.header
                    
                    # Bounding Box
                    bbox = BoundingBox2D()
                    bbox.center.x = float((xyxy[0] + xyxy[2]) / 2)
                    bbox.center.y = float((xyxy[1] + xyxy[3]) / 2)
                    bbox.size_x = float(xyxy[2] - xyxy[0])
                    bbox.size_y = float(xyxy[3] - xyxy[1])
                    detection.bbox = bbox

                    # Class ID and Score
                    detection.id = str(int(cls.item())) # Class ID as string
                    detection.score = float(conf.item())

                    detections_msg.detections.append(detection)

                    # 이미지에 바운딩 박스와 라벨 그리기
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle
                    label = f'{self.model.names[int(cls)]} {conf:.2f}' # Class name and confidence
                    cv2.putText(display_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 발행
        if camera_side == 'left':
            self.publisher_left.publish(detections_msg)
            self.detection_image_publisher_left.publish(self.bridge.cv2_to_imgmsg(display_image, "bgr8"))
        elif camera_side == 'right':
            self.publisher_right.publish(detections_msg)
            self.detection_image_publisher_right.publish(self.bridge.cv2_to_imgmsg(display_image, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
