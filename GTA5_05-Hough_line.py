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


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img=img, pt1=(coords[0], coords[1]), pt2=(coords[2], coords[3]), color=(255, 1, 1), thickness=3)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    processed_img = cv2.Canny(image=processed_img, threshold1=200, threshold2=300)

    vertices = np.array([[[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], ], np.int32)
    processed_img = roi(processed_img, vertices=vertices)
    processed_img = cv2.GaussianBlur(src=processed_img, ksize=(5, 5), sigmaX=0)
    lines = cv2.HoughLinesP(image=processed_img, rho=1, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=100, maxLineGap=5)
    # print(lines)
    '''
    [[[534 378 563 406]]
     [[735 315 773 328]]
     [[544 386 566 408]]
     ...
     [[146 262 161 248]]]
    '''
    draw_lines(processed_img, lines)

    return processed_img


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


if __name__ == '__main__':
    # screen_record_test()
    screen_record()
