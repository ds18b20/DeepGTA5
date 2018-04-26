#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageGrab
import cv2
import time
import functools
import json
from common.LineFunctions import *


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
    return wrapper


def dump_array(array: np.ndarray, fn='dump.txt'):
    with open(fn, 'w', encoding='utf8') as f:
        json.dump(array.tolist(), f)


def load_array(fn='dump.txt'):
    with open(fn, 'r', encoding='utf8') as f:
        return np.array(json.load(f))


class LaneFinder(object):
    def __init__(self, bound_box):
        self.__win_size = bound_box
        self.__raw_img = np.array(ImageGrab.grab(bbox=self.__win_size))
        self.__processed_img = self.img_process()

        # self.__lines = self.find_dummy_lines()
        self.__lines = self.find_lines()

        self.__lanes = self.find_lanes()
        self.__center = (400, 400)

    def get_lines(self):
        return self.__lines

    def get_lanes(self):
        return self.__lanes

    def get_processed_img(self):
        return self.__processed_img

    def get_center_coordinates(self):
        return self.__center

    def update(self):
        self.__raw_img = np.array(ImageGrab.grab(bbox=self.__win_size))
        self.__processed_img = self.img_process()
        # self.__lines = self.find_dummy_lines()
        self.__lines = self.find_lines()
        self.__lanes = self.find_lanes()

    def roi(self, img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(img, mask)
        return masked

    def img_process(self):
        # convert to gray
        temp_img = cv2.cvtColor(self.__raw_img, cv2.COLOR_BGR2GRAY)
        # blur
        temp_img = cv2.GaussianBlur(temp_img, ksize=(5, 5), sigmaX=0)
        # edge detection
        temp_img = cv2.Canny(temp_img, threshold1=100, threshold2=200)
        # region of interest
        vertices = np.array([[[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], ], np.int32)
        temp_img = self.roi(temp_img, vertices=vertices)
        # blur
        temp_img = cv2.GaussianBlur(src=temp_img, ksize=(5, 5), sigmaX=0)

        return temp_img

    def find_dummy_lines(self):
        return load_array(fn='dump.txt')

    def find_lines(self):
        # detect lines
        temp_lines = cv2.HoughLinesP(image=self.__processed_img,
                                     rho=1,
                                     theta=np.pi / 180,
                                     threshold=50,
                                     lines=np.array([]),
                                     minLineLength=75,
                                     maxLineGap=5)
        return temp_lines

    def find_lanes(self):
        # filter lines
        self.filter_lines()
        self.group_lines()
        left_lines, right_lines = self.seperate_lines_lr()
        return self.get_two_lanes(left_lines, right_lines)

    def filter_lines(self):
        # lines with slope > 0.25
        self.__lines = np.array(list(filter(lambda i: abs(slope_of_line(i)) > 0.25, self.__lines)))

    def group_lines(self):
        lines_stay = self.__lines
        lane_group = []
        # count = 0
        while len(lines_stay) > 0:
            # if count > 100:
            #     break
            # count += 1
            del_list = []
            line_temp = lines_stay[0]
            lines_stay = np.delete(lines_stay, 0, axis=0)
            for index in range(len(lines_stay)):
                # print('Count: {}'.format(count-1))
                # print('Index: {}'.format(index))
                # print('Diff: {}'.format(line_diff(lines_stay[index], line_temp)))
                # print('Left: {}'.format(lines_stay))
                if line_diff(lines_stay[index], line_temp):
                    del_list.append(index)
                    line_temp = np.concatenate([line_temp, lines_stay[index]], axis=0)
            lines_stay = np.delete(lines_stay, del_list, axis=0)
            lane_group.append(line_temp.reshape(-1, 1, 4))
        self.__lanes = np.array([np.average(lg, axis=0) for lg in lane_group]).astype(np.int)

    # filter lanes to left and right lanes
    def seperate_lines_lr(self):
        # center = self.__CENTER
        lanes_l = []
        lanes_r = []

        for lane_lr in self.__lanes:
            if slope_of_line(lane_lr) > 0:
                lanes_l.append(lane_lr)
            elif slope_of_line(lane_lr) < 0:
                lanes_r.append(lane_lr)
            else:
                print('error')
        return np.array(lanes_l), np.array(lanes_r)

    def get_two_lanes(self, left_lanes, right_lanes):
        left_sorted_list = sorted(left_lanes, key=self.sorted_key)
        left_lane_list = left_sorted_list[0]

        right_sorted_list = sorted(right_lanes, key=self.sorted_key)
        right_lane_list = right_sorted_list[0]

        ret = np.array([left_lane_list, right_lane_list])
        return ret

    def sorted_key(self, key_x):
        # return distance_point_line(self.__center, key_x)

        return distance_point_line((400, 400), key_x)

    def draw_lines(self, lines: np.ndarray):
        # try:
        #     for line in lines:
        #         coords = line[0]
        #         cv2.line(img=img, pt1=(coords[0], coords[1]), pt2=(coords[2], coords[3]), color=(255, 1, 1), thickness=3)
        # except:
        #     pass
        # for index, line in enumerate(self.__lines):
        for index, line in enumerate(lines):
            if lines.ndim == 3:
                line = line[0]
            cv2.line(img=self.__processed_img, pt1=(line[0], line[1]), pt2=(line[2], line[3]), color=(255, 1, 1),
                     thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.__processed_img, str(line), (line[0], line[1] - 10), font, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    def draw_text(self, text: str, coordinates: tuple):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.__processed_img, str(text), (coordinates[0], coordinates[1]), font, 2, (255, 255, 255), 1,
                    cv2.LINE_AA)

    def show_img(self):
        cv2.imshow('window', self.__processed_img)


def dummy_test():
    window = (0, 40, 800, 640)  # 800*600 from up-left
    lanes = LaneFinder(window)
    print('lines here:')
    dummy_lines = lanes.get_lines()
    print(dummy_lines)

    print('lanes here:')
    dummy_lanes = lanes.get_lanes()
    print(dummy_lanes.shape)
    print(dummy_lanes)

    print(slope_of_line(np.array([[100, 300, 400, 302]])))


def run():
    window = (0, 40, 800, 640)  # 800*600 from up-left
    lanes = LaneFinder(window)
    while True:
        try:
            lanes.update()
        except:
            pass
        print('lanes')
        for lane in lanes.get_lanes():
            print(lane)
            lanes.draw_lines(lane)
            lanes.draw_text('.', lanes.get_center_coordinates())
            center_x, center_y = lanes.get_center_coordinates()
            try:
                feet_x, feet_y = [int(_) for _ in perpendicular_feet(lanes.get_center_coordinates(), lane)]
                lanes.draw_lines(np.array([[center_x, center_y, feet_x, feet_y]]))
            except:
                pass
        lanes.show_img()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    run()
