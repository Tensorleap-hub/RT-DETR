# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Source: https://huggingface.co/spaces/motaer0206/YOLOv12-Audio-Assistant/blob/1999a98cdd42d8ab98824612ca47b96a94ada048/examples/RTDETR-ONNXRuntime-Python/main.py

import argparse
import json
import traceback

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from ultralytics.utils import ASSETS
from ultralytics.utils.checks import check_requirements, check_yaml


class RTDETR:
    """RTDETR object detection model class for handling inference and visualization."""

    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.5):
        """
        Implements pre and postprocessing for rtdetr like onnx models.
        Args:
            model_path: Path to the ONNX model file.
            img_path: Path to the input image.
            conf_thres: Confidence threshold for object detection.
            iou_thres: IoU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.img_path = ""
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Set up the ONNX runtime session with CUDA and CPU execution providers
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])#, "CPUExecutionProvider"])
        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[3]
        self.input_height = self.model_input[0].shape[2]

        # Load class names from the COCO dataset YAML file
        self.classes = [
            "Person",
            "Vehicle",
            "Armored_Vehicle",
        ]

        # Generate a color palette for drawing bounding boxes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def update_img_path(self, img_path):
        """
        Updates the input image path.
        Args:
            img_path: Path to the new input image.
        Returns:
            None
        """
        self.img_path = img_path

    def draw_detections(self, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.
        Args:
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.
        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            self.img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(
            self.img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.
        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(str(self.img_path))

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes):
        """
        Converts bounding boxes from (center x, center y, width, height) format to (x_min, y_min, x_max, y_max) format.
        Args:
            boxes (numpy.ndarray): An array of shape (N, 4) where each row represents
                                a bounding box in (cx, cy, w, h) format.
        Returns:
            numpy.ndarray: An array of shape (N, 4) where each row represents
                        a bounding box in (x_min, y_min, x_max, y_max) format.
        """
        # Calculate half width and half height of the bounding boxes
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2

        # Calculate the coordinates of the bounding boxes
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height

        # Return the bounding boxes in (x_min, y_min, x_max, y_max) format
        return np.column_stack((x_min, y_min, x_max, y_max))

    def process_model_output(self, model_output: np.ndarray | list) -> tuple:
        # For rtdetr, the model output can either be:
        # - tuple of (boxes, scores, logits)
        # - tuple of (boxes, scores)
        # - tuple of (num detections, detection_boxes, detection_scores, detection_classes)
        boxes = None
        scores = None
        logits = None
        labels = None

        if len(model_output) <=3:
            # Output format:
            # 0: boxes
            # 1: scores
            # 2: Optional: logits
            boxes = model_output[0]
            scores = model_output[1]
            logits = model_output[2]

            # Get detection_classes from scores:
            num_classes = scores.shape[2]
            assert num_classes == len(self.classes), f"Expected {len(self.classes)} classes, but got {num_classes}"
            labels = np.argmax(scores, axis=2)

            # Get max confidence scores
            scores = np.max(scores, axis=2)

            # Convert boxes from (center x, center y, width, height) to (x_min, y_min, x_max, y_max)
            for i in range(boxes.shape[0]):
                boxes[i] = self.bbox_cxcywh_to_xyxy(boxes[i])

            scale_factor_x = self.img_width / self.input_width
            scale_factor_y = self.img_height / self.input_height
            boxes[:, :, 0::2] *= scale_factor_x #self.img_width
            boxes[:, :, 1::2] *= scale_factor_y #self.img_height

            if len(model_output) == 3:
                logits = model_output[2]
        elif len(model_output) == 4:
            # Output format:
            # 0: num detections
            # 1: detection_boxes
            # 2: detection_scores
            # 3: detection_classes
            boxes = model_output[1]
            scores = model_output[2]
            labels = model_output[3]

            # Scale bounding boxes to match the original image dimensions
            scale_faxtor_x = self.img_width / self.input_width
            scale_factor_y = self.img_height / self.input_height
            boxes[:, 0::2] *= scale_faxtor_x #self.img_width
            boxes[:, 1::2] *= scale_factor_y #self.img_height
        else:
            raise ValueError(f"Unexpected model output format: {len(model_output)}")

        return boxes, scores, labels, logits


    def postprocess(self, model_output):
        """
        Postprocesses the model output to extract detections and draw them on the input image.
        Args:
            model_output: Output of the model inference.
        Returns:
            np.array: Annotated image with detections.
        """
        # Process the model output
        boxes, scores, labels, logits = self.process_model_output(model_output)

        # Apply confidence threshold to filter out low-confidence detections
        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        # debug: 
        # self.img = cv2.resize(self.img, (self.input_width, self.input_height))

        # Draw detections on the image
        for box, score, label in zip(boxes, scores, labels):
            self.draw_detections(box, score, label)

        # Return the annotated image
        return self.img, (boxes, scores, labels)

    def main(self):
        """
        Executes the detection on the input image using the ONNX model.
        Returns:
            np.array: Output image with annotations.
        """
        # Preprocess the image for model input
        image_data = self.preprocess()

        # Run the model inference
        model_output = self.session.run(None, {self.model_input[0].name: image_data})

        # Process and return the model output
        return self.postprocess(model_output)


if __name__ == "__main__":
    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rtdetr-l.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to the input image.")
    parser.add_argument("--conf-thres", type=float, default=0.01, help="Confidence threshold for object detection.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.")
    parser.add_argument("--input-dir", type=str, default=str("./input"), help="Path to the input directory.")
    parser.add_argument("--output-dir", type=str, default=str("./output"), help="Path to the output directory.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualizing of all images after inference")
    args = parser.parse_args()

    # Check for dependencies and set up ONNX runtime
    # check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # Search all images in input dir
    logger.info(f"Input directory: {args.input_dir}")
    input_dir = Path(args.input_dir)
    img_paths = list(input_dir.glob("*"))
    logger.info(f"Found {len(img_paths)} images in the input directory.")

    # Create the output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Create the detector instance with specified parameters
    detection = RTDETR(args.model, args.conf_thres, args.iou_thres)

    for img_path in tqdm(img_paths):
        # update image path
        img_path = Path(img_path)
        detection.img_path = img_path

        # Perform detection and get the output image
        try:
            output_image, results = detection.main()
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}"
                         f"\n{traceback.format_exc()}")
            logger.info(f"Skipping image {img_path}.")
            continue

        # Save output boxes as a JSON file
        json_file_name = img_path.stem + ".json"
        json_path = Path(args.output_dir) / json_file_name
        boxes = results[0]
        scores = results[1]
        labels = results[2]

        with open(json_path, "w") as f:
            json.dump({"boxes": boxes.tolist(), "scores": scores.tolist(), "labels": labels.tolist()}, f)

        if args.visualize and output_image is not None:
            # Display the annotated output image
            cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Output", output_image)
            cv2.waitKey(0)

    logger.success(f"Done!")