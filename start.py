#!/usr/bin/env python

# Bird Feeder - Feed Birds & Capture Images!
# Copyright (C) 2020 redlogo
#
# This program is under MIT license

import time
from datetime import datetime

import cv2

from rpi.rpi_camera import RPiCamera
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
    print('RPi Bird Feeder -> RPi Camera Ready')
    time.sleep(1.0)

    # initialize DSLR
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
    callback_obj = gp.check_result(gp.use_python_logging())
    DSLR = gp.Camera()
    DSLR.init()
    print('RPi Bird Feeder -> DSLR Ready')
    time.sleep(0.5)

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
    bird_time = time.monotonic()
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
        after_render_time = time.monotonic()
        for i in range(len(class_ids)):
            if int(class_ids[i]) == 15 and (after_render_time - bird_time) > 5:
                bird_time = time.monotonic()
                DSLR_path = DSLR.capture(gp.GP_CAPTURE_IMAGE)
                print(datetime.now())
                target = os.path.join('./DSLR', str(datetime.now()) + '.jpg')
                DSLR_file = DSLR.file_get(DSLR_path.folder, DSLR_path.name, gp.GP_FILE_TYPE_NORMAL)
                DSLR_file.save(target)
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
                     moving_average_image_show_time.get_moving_average() / total_time * 100))#, end='\r')

        # counter
        image_count += 1
        if image_count == 10000000:
            image_count = 0


if __name__ == "__main__":
    main()
