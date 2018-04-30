#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageGrab
import cv2
import time
import functools
import json
from common.LineFunctions import *
from common.directkeys import *


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
        self.__CENTER = (400, 400)
        # get source image
        self.__win_size = bound_box
        self.__raw_img = np.array(ImageGrab.grab(bbox=self.__win_size))
        # self.__raw_img = cv2.imread('2b_lines.jpg', flags=cv2.IMREAD_COLOR)

        # color 2gray & get region of interest
        self.__processed_img = self.img_process()

        # find raw lines
        self.__raw_lines = self.find_lines()
        # self.__lines = self.find_dummy_lines()

        # filter lines
        self.__filtered_lines = self.filter_lines()

        # group lines with similar slope
        # average each group
        self.__group_of_lines_avr = self.group_lines_avr()

        # separate lines 2 a tuple of left lines and right lines
        # return as a tuple
        self.__tuple_of_lines_lr = self.separate_lines_lr()

        # find one line on each side as lanes
        # return as a tuple
        self.__lanes = self.get_two_lanes()

    def get_raw_lines(self):
        return self.__raw_lines

    def get_lanes(self):
        return self.__lanes

    def get_processed_img(self):
        return self.__processed_img

    def get_center_coordinates(self):
        return self.__CENTER

    def update(self):
        self.__raw_img = np.array(ImageGrab.grab(bbox=self.__win_size))
        # self.__raw_img = cv2.imread('2b_lines.jpg', flags=cv2.IMREAD_COLOR)

        # color 2gray & get region of interest
        self.__processed_img = self.img_process()

        # find raw lines
        self.__raw_lines = self.find_lines()
        # self.__lines = self.find_dummy_lines()

        # filter lines
        self.__filtered_lines = self.filter_lines()

        # group lines with similar slope
        # average each group
        self.__group_of_lines_avr = self.group_lines_avr()

        # separate lines 2 a tuple of left lines and right lines
        # return as a tuple
        self.__tuple_of_lines_lr = self.separate_lines_lr()

        # find one line on each side as lanes
        # return as a tuple
        self.__lanes = self.get_two_lanes()

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
        # vertices = np.array([[[10, 590], [10, 400], [300, 300], [500, 300], [790, 400], [790, 590]], ], np.int32)
        # vertices = np.array([[[10, 500], [10, 300], [300, 200], [500, 200], [800, 300], [800, 500]], ], np.int32)
        vertices = np.array([[[10, 500], [10, 400], [300, 300], [500, 300], [800, 400], [800, 500]], ], np.int32)
        # vertices = np.array([[[10, 450], [10, 335], [260, 260], [470, 260], [790, 335], [790, 450], [500, 450], [445, 320], [335, 320], [275, 450]]], np.int32)
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

    def filter_lines(self):
        # lines with abs(slope) > 0.25
        if isinstance(self.__raw_lines, np.ndarray):
            temp_lines = np.array(list(filter(lambda i: i[0][0] > 100 and i[0][1] > 300, self.__raw_lines)))
            temp_lines = np.array(list(filter(lambda i: abs(slope_of_line(i)) > 0.25, temp_lines)))

        else:
            temp_lines = None
        return temp_lines

    def group_lines_avr(self):
        if isinstance(self.__filtered_lines, np.ndarray):
            lines_stay = self.__filtered_lines
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
            temp_lines = np.array([np.average(lg, axis=0) for lg in lane_group]).astype(np.int)
        else:
            temp_lines = None
        return temp_lines

    # filter lanes to left and right lanes
    def separate_lines_lr(self):
        if isinstance(self.__group_of_lines_avr, np.ndarray):
            lines_l = []
            lines_r = []
            for lane_lr in self.__group_of_lines_avr:
                # 注意此处角度正负和正常坐标系相反
                # 因为图像左上角是原点，右侧为x正，下侧为y正
                if slope_of_line(lane_lr) < 0:
                    lines_l.append(lane_lr)
                elif slope_of_line(lane_lr) > 0:
                    lines_r.append(lane_lr)
                else:
                    print('line slope error!')
            if len(lines_l):
                ret_l = np.array(lines_l)
            else:
                ret_l = None
            if len(lines_r):
                ret_r = np.array(lines_r)
            else:
                ret_r = None

            return ret_l, ret_r
        else:
            return None, None

    def get_two_lanes(self):
        # left lane
        left_side_lines = self.__tuple_of_lines_lr[0]
        if isinstance(left_side_lines, np.ndarray):
            left_sorted_list = sorted(left_side_lines, key=self.sorted_key)
            left_lane = left_sorted_list[0]
        else:
            left_lane = None

        # right lane
        right_side_lines = self.__tuple_of_lines_lr[1]
        if isinstance(right_side_lines, np.ndarray):
            right_sorted_list = sorted(right_side_lines, key=self.sorted_key)
            right_lane = right_sorted_list[0]
        else:
            right_lane = None

        return left_lane, right_lane

    def sorted_key(self, key_x):
        # return distance_point_line(self.__center, key_x)
        return distance_point_line(self.__CENTER, key_x)

    def draw_lines(self, lines: np.ndarray):
        # try:
        #     for line in lines:
        #         coords = line[0]
        #         cv2.line(img=img, pt1=(coords[0], coords[1]), pt2=(coords[2], coords[3]), color=(255, 1, 1), thickness=3)
        # except:
        #     pass
        # for index, line in enumerate(self.__lines):
        if isinstance(lines, np.ndarray):
            for index, line in enumerate(lines):
                cv2.line(img=self.__processed_img,
                         pt1=(line[0], line[1]),
                         pt2=(line[2], line[3]),
                         color=(255, 1, 1),
                         thickness=3)
                cv2.putText(img=self.__processed_img,
                            text=str(line),
                            org=(line[0], line[1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 255, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA)
            # print('Draw lane OK')
        else:
            print('No lanes to draw!')

    def draw_text(self, text: str, coordinates: tuple):
        cv2.putText(img=self.__processed_img,
                    text=text,
                    org=(coordinates[0], coordinates[1] + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    def show_img(self):
        cv2.imshow('window', self.__processed_img)


def dummy_test():
    window = (0, 40, 800, 640)  # 800*600 from up-left
    lanes = LaneFinder(window)
    print('lines here:')
    dummy_lines = lanes.get_raw_lines()
    print(dummy_lines)

    print('lanes here:')
    dummy_lanes = lanes.get_lanes()
    print(dummy_lanes)

    print(slope_of_line(np.array([[100, 300, 400, 302]])))


def straight():
    PressKey(W)

    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(A)
    PressKey(W)

    ReleaseKey(D)


def right():
    PressKey(D)
    PressKey(W)

    ReleaseKey(A)


def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def run_new():
    for i in range(6):
        time.sleep(1)
        print(6 - i)
    window = (0, 20, 800, 620)  # 800*600 from up-left
    lanes = LaneFinder(window)
    while True:
        las_time = time.time()
        lanes.update()
        lane_lr = lanes.get_lanes()

        # left lane
        if isinstance(lane_lr[0], np.ndarray):
            distance_l = abs(distance_point_line((400, 400), lane_lr[0]))
            # slope_l = slope_of_line(lane_lr[0])

        else:
            distance_l = None
            # slope_l = None

        # right lane
        if isinstance(lane_lr[1], np.ndarray):
            distance_r = abs(distance_point_line((400, 400), lane_lr[1]))
            # slope_r = slope_of_line(lane_lr[1])
        else:
            distance_r = None
            # slope_r = None

        if distance_l and distance_r:
            if distance_l > 80 and distance_r > 80:
                straight()
            elif distance_l < 80:
                right()
                print('Left line-Left')
            elif distance_r < 80:
                left()
                print('Right line-Left')
            else:
                print('impossible')

        elif distance_r:
            if distance_r < 75:
                left()
                print('Right line-Left')
            elif distance_r > 120:
                right()
                print('Right line-Right')
            else:
                straight()
                print('Right line-Straight')

        elif distance_l:
            if distance_l < 75:
                right()
                print('Left line-Left')
            elif distance_l > 125:
                left()
                print('Left line-Right')
            else:
                straight()
                print('Left line-Straight')
        else:
            slow_ya_roll()
            print('Slow')

        for lane in lanes.get_lanes():
            # print(lane)
            lanes.draw_lines(lane)
            # lanes.draw_text('.', lanes.get_center_coordinates())
            # center_x, center_y = lanes.get_center_coordinates()
            # try:
            #     feet_x, feet_y = [int(_) for _ in perpendicular_feet(lanes.get_center_coordinates(), lane)]
            #     lanes.draw_lines(np.array([[center_x, center_y, feet_x, feet_y]]))
            # except:
            #     pass
        lanes.show_img()
        print(time.time() - las_time)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    run_new()
