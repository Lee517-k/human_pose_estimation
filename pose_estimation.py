#!/usr/bin/env python3
import collections
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)
with open("notebook_utils.py", "w") as f:
    f.write(r.text)

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/engine3js.py",
)
with open("engine3js.py", "w") as f:
    f.write(r.text)

import notebook_utils as utils
import engine3js as engine

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'rgb_left',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'pose_estimation', 10)
        self.subscription  # prevent unused variable warning

        # Model initialization code
        base_model_dir = "model"
        model_name = "human-pose-estimation-3d-0001"
        precision = "FP32"
        model_path = Path(f"{base_model_dir}/public/{model_name}/{model_name}").with_suffix(".pth")
        onnx_path = Path(f"{base_model_dir}/public/{model_name}/{model_name}").with_suffix(".onnx")

        ir_model_path = f"model/public/{model_name}/{precision}/{model_name}.xml"
        model_weights_path = f"model/public/{model_name}/{precision}/{model_name}.bin"

        if not model_path.exists():
            download_command = f"omz_downloader --name {model_name} --output_dir {base_model_dir}"
            get_ipython().system(' $download_command')

        if not onnx_path.exists():
            convert_command = f"omz_converter --name {model_name} --precisions {precision} --download_dir {base_model_dir} --output_dir {base_model_dir}"
            get_ipython().system(' $convert_command')

        core = ov.Core()
        model = core.read_model(model=ir_model_path, weights=model_weights_path)
        self.compiled_model = core.compile_model(model=model, device_name="CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor_name = model.inputs[0].get_any_name()

        self.processing_times = collections.deque()
        self.focal_length = -1
        self.stride = 8

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        scaled_img = cv2.resize(frame, dsize=(self.compiled_model.inputs[0].shape[3], self.compiled_model.inputs[0].shape[2]))

        if self.focal_length < 0:
            self.focal_length = np.float32(0.8 * scaled_img.shape[1])

        start_time = time.time()
        results = self.model_infer(scaled_img)
        stop_time = time.time()
        self.processing_times.append(stop_time - start_time)

        poses_3d, poses_2d = engine.parse_poses(results, 1, self.stride, self.focal_length, True)
        frame = self.draw_poses(frame, poses_2d, scaled_img)

        self.publisher.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

        if len(self.processing_times) > 200:
            self.processing_times.popleft()

    def model_infer(self, scaled_img):
        img = scaled_img[0 : scaled_img.shape[0] - (scaled_img.shape[0] % self.stride), 0 : scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]
        img = np.transpose(img, (2, 0, 1))[None,]
        self.infer_request.infer({self.input_tensor_name: img})
        results = {name: self.infer_request.get_tensor(name).data[:] for name in {"features", "heatmaps", "pafs"}}
        return results["features"][0], results["heatmaps"][0], results["pafs"][0]

    def draw_poses(self, frame, poses_2d, scaled_img):
        body_edges_2d = np.array(
            [[0, 1], [1, 16], [16, 18], [1, 15], [15, 17], [0, 3], [3, 4], [4, 5],
             [0, 9], [9, 10], [10, 11], [0, 6], [6, 7], [7, 8], [0, 12], [12, 13], [13, 14]]
        )
        for pose in poses_2d:
            pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
            was_found = pose[2] > 0
            pose[0], pose[1] = pose[0] * frame.shape[1] / scaled_img.shape[1], pose[1] * frame.shape[0] / scaled_img.shape[0]
            for edge in body_edges_2d:
                if was_found[edge[0]] and was_found[edge[1]]:
                    cv2.line(frame, tuple(pose[0:2, edge[0]].astype(np.int32)), tuple(pose[0:2, edge[1]].astype(np.int32)), (255, 255, 0), 4, cv2.LINE_AA)
            for kpt_id in range(pose.shape[1]):
                if pose[2, kpt_id] != -1:
                    cv2.circle(frame, tuple(pose[0:2, kpt_id].astype(np.int32)), 3, (0, 255, 255), -1, cv2.LINE_AA)
        return frame


def main(args=None):
    rclpy.init(args=args)
    pose_estimation_node = PoseEstimationNode()
    rclpy.spin(pose_estimation_node)
    pose_estimation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
