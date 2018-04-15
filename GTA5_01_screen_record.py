#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageGrab
import cv2
import time
import functools


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
    return wrapper


def screen_record():
    while True:
        print_screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
        cv2.imshow('window', cv2.cvtColor(print_screen, cv2.COLOR_BGR2RGB))
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
