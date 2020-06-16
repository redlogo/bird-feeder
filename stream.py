import time

import cv2

from sender.rpi_camera import RPiCamera
from object_detection.object_detection import Model
from utilities.render import Render
from utilities.stats import MovingAverage

import logging
import os
import subprocess
import sys
import gphoto2 as gp


def main():
    """
    main function interface
    :return: nothing
    """

    # params
    width = 640
    height = 368
    moving_average_points = 50

    # initialize RPi camera
    rpi_cam = RPiCamera(width, height)
    rpi_cam.start()
    print('RPi Bird Feeder -> Camera Started')
    time.sleep(1.0)

    # initialize object detection model
    model = Model()
    model.load_model('models_edgetpu/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    model.load_labels('labels_edgetpu/coco_labels.txt')
    model.set_confidence_level(0.6)
    print('RPi Bird Feeder -> Object Detection Model Initialized')
    time.sleep(1.0)

    # initialize render
    render = Render()
    print('RPi Bird Feeder -> Render Ready')
    time.sleep(0.5)

    # statistics
    moving_average_fps = MovingAverage(moving_average_points)
    moving_average_receive_time = MovingAverage(moving_average_points)
    moving_average_model_load_image_time = MovingAverage(moving_average_points)
    moving_average_model_inference_time = MovingAverage(moving_average_points)
    moving_average_image_show_time = MovingAverage(moving_average_points)

    # streaming
    image_count = 0
    print('RPi Bird Feeder -> Receiver Streaming')
    time.sleep(0.5)
    while True:
        start_time = time.monotonic()

        # get image
        image = rpi_cam.get_image()
        received_time = time.monotonic()

        # load image into model (cv2 or pil backend)
        model.load_image_cv2_backend(image)
        model_loaded_image_time = time.monotonic()

        # model inference
        class_ids, scores, boxes = model.inference()
        model_inferenced_time = time.monotonic()

        # render image
        render.set_image(image)
        render.render_detection(model.labels, class_ids, boxes, image.shape[1], image.shape[0], (45, 227, 227), 3)
        print(class_ids)
        render.render_fps(moving_average_fps.get_moving_average())

        # show image
        cv2.imshow('Bird Feeder', image)
        image_showed_time = time.monotonic()
        if cv2.waitKey(1) == ord('q'):
            break

        # statistics
        instant_fps = 1 / (image_showed_time - start_time)
        moving_average_fps.add(instant_fps)
        receive_time = received_time - start_time
        moving_average_receive_time.add(receive_time)
        model_load_image_time = model_loaded_image_time - received_time
        moving_average_model_load_image_time.add(model_load_image_time)
        model_inference_time = model_inferenced_time - model_loaded_image_time
        moving_average_model_inference_time.add(model_inference_time)
        image_show_time = image_showed_time - model_inferenced_time
        moving_average_image_show_time.add(image_show_time)
        total_time = moving_average_receive_time.get_moving_average() \
                     + moving_average_model_load_image_time.get_moving_average() \
                     + moving_average_model_inference_time.get_moving_average() \
                     + moving_average_image_show_time.get_moving_average()

        # terminal prints
        if image_count % 50 == 0:
            print(" receiver's fps: %4.1f"
                  " receiver's time components: "
                  "receiving %4.1f%% "
                  "model load image %4.1f%% "
                  "model inference %4.1f%% "
                  "image show %4.1f%%"
                  % (moving_average_fps.get_moving_average(),
                     moving_average_receive_time.get_moving_average() / total_time * 100,
                     moving_average_model_load_image_time.get_moving_average() / total_time * 100,
                     moving_average_model_inference_time.get_moving_average() / total_time * 100,
                     moving_average_image_show_time.get_moving_average() / total_time * 100), end='\r')

        # counter
        image_count += 1
        if image_count == 10000000:
            image_count = 0


if __name__ == "__main__":
    main()
