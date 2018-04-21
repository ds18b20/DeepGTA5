#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageGrab
import cv2
import time
import functools
# import pyautogui


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
    return wrapper


def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], ], np.int32)
    processed_img = roi(processed_img, vertices=vertices)
    return processed_img


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def screen_record():
    while True:
        last_time = time.time()
        print_screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        new_screen = process_img(print_screen)
        print('Loop takes {} s'.format('%.4f' % (time.time() - last_time)))

        cv2.imshow('window', new_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


@timeit
def screen_record_test():
    print_screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    # print(print_screen_pil.size)
    # screen_numpy = np.array(screen_pil.getdata(), dtype=np.uint8)
    # screen_numpy = screen_numpy.reshape(screen_pil.size[1], screen_pil.size[0], -1)
    # print(screen_numpy.shape)
    cv2.imshow('window', print_screen)


if __name__ == '__main__':
    # screen_record_test()
    screen_record()
