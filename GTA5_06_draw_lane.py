#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageGrab
import cv2
import time
import functools
import json
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
    processed_img = cv2.Canny(image=processed_img, threshold1=100, threshold2=200)

    vertices = np.array([[[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], ], np.int32)
    processed_img = roi(processed_img, vertices=vertices)
    processed_img = cv2.GaussianBlur(src=processed_img, ksize=(5, 5), sigmaX=0)
    lines = cv2.HoughLinesP(image=processed_img, rho=1, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=100, maxLineGap=5)
    with open('dump.txt', 'w', encoding='utf8') as f:
        json.dump(lines.tolist(), f)
    # print(type(lines))
    '''
    [[[534 378 563 406]]
     [[735 315 773 328]]
     [[544 386 566 408]]
     ...
     [[146 262 161 248]]]
    '''
    draw_lines(processed_img, lines)
    # draw_lanes(processed_img, lines)

    return processed_img


def value_of_line(x, y, line: np.ndarray):
    delta = 1e-6
    if line.ndim == 2:
        line = line[0]
    if line[3] - line[1] < delta:
        return y - line[3]
    elif line[2] - line[0] < delta:
        return x - line[2]
    else:
        return (y - line[1]) / (line[3] - line[1]) - (x - line[0]) / (line[2] - line[0])


def line_diff(line: np.ndarray, line_base: np.ndarray):
    if line.ndim == 2:
        line = line[0]
    return value_of_line(line[0], line[1], line_base)**2 + value_of_line(line[2], line[3], line_base)**2


def length_of_line(line: np.ndarray):
    if line.ndim == 2:
        line = line[0]
    return ((line[2] - line[0]) ** 2 + (line[3] - line[1]) ** 2) ** (1 / 2)


def slope_of_line(line: np.ndarray):
    delta = 1e-6
    if line.ndim == 2:
        line = line[0]
    return (line[3] - line[1]) / (line[2] - line[0] + delta)


def find_lanes(lines: np.ndarray, min_length=100, error=100):
    min_length = 200
    lines = np.array(list(filter(lambda x: length_of_line(x) > min_length, lines)))

    line_group = []
    line_left = np.copy(lines)
    count = 0
    while len(line_left) > 0:
        if count > 1000:
            break
        count += 1
        del_list = []

        line_temp = line_left[0]
        line_left = np.delete(line_left, 0, axis=0)
        # print(line_left)
        for index in range(len(line_left)):
            # print(count-1, index, line_diff(line_left[index], line_temp))
            if line_diff(line_left[index], line_temp) < error:
                del_list.append(index)
                line_temp = np.concatenate([line_temp, line_left[index]], axis=0)
        line_left = np.delete(line_left, del_list, axis=0)
        line_group.append(line_temp.reshape(-1, 1, 4))

    return np.array([np.average(lg, axis=0) for lg in line_group])


def draw_lanes(img, lines):
    temp = find_lanes(lines)
    print(temp)
    draw_lines(img, temp.astype(np.int8))


def draw_lines(img, lines):
    # try:
    #     for line in lines:
    #         coords = line[0]
    #         cv2.line(img=img, pt1=(coords[0], coords[1]), pt2=(coords[2], coords[3]), color=(255, 1, 1), thickness=3)
    # except:
    #     pass

    for line in lines:
        coords = line[0]
        cv2.line(img=img, pt1=(coords[0], coords[1]), pt2=(coords[2], coords[3]), color=(255, 1, 1), thickness=3)


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
    # print(length_of_line([0, 0, 1, 1]))
    # a = np.array([[[0, 0, 1, 1]], [[0, 0, 2, 2]], [[0, 0, 3, 3]]])
    # print(list(find_lane(a)))
    # print(value_of_line(0, 2, a[0]))
    # lines = np.array(filter(lambda x: length_of_line(x) > 2, a))
    # print(lines)
