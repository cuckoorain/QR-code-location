import numpy as np
import cv2
import os
from my_location import DistanceCalculate_forLine
from my_location import LeftClikcShow
import math

PI = 3.1415926535


def RotateAngle_z(focal_length, real_target_line_length, real_distance, seen_pixel_target_line):
    """
    根据透视原理求得target线段关于z轴的旋转角度，其中预设assistant线段与z轴平行，assistant与target构成的平面与z轴平行
    :param focal_length: 焦距
    :param real_target_line_length: target线段物理长度
    :param real_distance:摄像头到assistant线段的实际长度
    :param seen_pixel_target_line:target线段在图像中的长度
    :return:assistant与target构成的平面绕z轴的旋转角度
    """

    should_see_pixel_target_line = (real_target_line_length * focal_length) / real_distance
    alpha = math.acos(1.0 * seen_pixel_target_line / should_see_pixel_target_line) / 3.1415926 * 180
    # alpha = math.acos(1.0 * should_see_pixel_target_line / seen_pixel_target_line) / 3.1415926 * 180
    return alpha


def RotateAngle_z_cosMethod(focal_length, real_height, real_width, distance_target_line_m, distance_target_line_n, pixel_x_m, if_mn_same_side):
    r_m = distance_target_line_m
    r_n = distance_target_line_n
    r_m_n = real_width
    gamma = (r_m ** 2 + r_m_n ** 2 - r_n ** 2) / (2 * r_m * r_m_n)

    pixel_r_m = real_height * focal_length / r_m
    if if_mn_same_side > 0:
        # m和n在同一侧
        beta = 180 - math.acos(pixel_x_m / pixel_r_m) / PI * 180
    if if_mn_same_side < 0:
        # m和n不在同一侧
        beta = math.acos(pixel_x_m / pixel_r_m) / PI * 180
    if pixel_x_m == 0:
        beta = 90
    alpha = math.acos(gamma) / PI * 180 - beta
    return alpha


def RotateAngle_z_cosMethod_main():
    main_path = str(r'D:\my_Academic\teacher_zyb\test_angle')
    # main_path = str(r'D:\my_Academic\teacher_zyb\test_angle_backup0')
    KNOWN_DISTANCE = 50
    KNOWN_WIDTH = 29.7
    KNOWN_HEIGHT = 21
    if_retest_focal_flag_focal_length = 0
    while (1):
        if if_retest_focal_flag_focal_length == 0:
            if_retest_focal = 1
        else:
            if_retest_focal = 0
        real_distance_m, focal_length, x_m_pixel = DistanceCalculate_forLine(main_path=main_path, if_retest_focal=if_retest_focal,
                                                                  focal_length_input=if_retest_focal_flag_focal_length, if_return_x = 1)
        if_retest_focal_flag_focal_length = focal_length
        real_distance_n, _, x_n_pixel = DistanceCalculate_forLine(main_path=main_path, if_retest_focal=0,
                                                       focal_length_input=if_retest_focal_flag_focal_length, if_return_x = 1)

        # distance = DistanceCalculate(main_path=main_path)
        print("calculate by assistant line for real_distance_m is {}".format(real_distance_m))
        print("calculate by assistant line for real_distance_n is {}".format(real_distance_n))
        print("x_n_pixel is {}".format(x_n_pixel))
        print("x_m_pixel is {}".format(x_m_pixel))
        if x_n_pixel * x_m_pixel > 0:
            if_mn_same_side_flag = 1
        elif x_n_pixel * x_m_pixel < 0:
            if_mn_same_side_flag = -1
        elif x_n_pixel == 0:
            if_mn_same_side_flag = 0
        else:
            print("wrong in detecting x_n and x_m")

        alpha = RotateAngle_z_cosMethod(focal_length = focal_length, real_height = KNOWN_HEIGHT, real_width=KNOWN_WIDTH, distance_target_line_m=real_distance_m,
                                        distance_target_line_n=real_distance_n, pixel_x_m=x_m_pixel, if_mn_same_side= if_mn_same_side_flag)
        print("alpha is {}".format(alpha))


if __name__ == "__main__":

    RotateAngle_z_cosMethod_main()
    #
    # PI = 3.1415926535
    # main_path = str(r'D:\my_Academic\teacher_zyb\test_angle')
    # KNOWN_DISTANCE = 50
    # KNOWN_WIDTH = 29.7
    # KNOWN_HEIGHT = 21
    # if_retest_focal_flag_focal_length = 0
    # while (1):
    #     if if_retest_focal_flag_focal_length == 0:
    #         if_retest_focal = 1
    #     else:
    #         if_retest_focal = 0
    #     real_distance, focal_length = DistanceCalculate_forLine(main_path=main_path, if_retest_focal=if_retest_focal,
    #                                                             focal_length_input=if_retest_focal_flag_focal_length)
    #     if_retest_focal_flag_focal_length = focal_length
    #     # distance = DistanceCalculate(main_path=main_path)
    #     print("calculate by assistant line for distance is {}".format(real_distance))
    #     img_name_input = input("test_name_input e.g. 50cm30c")
    #     img_name_input = img_name_input + ".jpg"
    #     img_file_angle = os.path.join(main_path, img_name_input)
    #     img_angle = cv2.imread(img_file_angle)
    #     print("rotation  Angle_test_final shape is {}".format(img_angle.shape))
    #     points = LeftClikcShow(img_angle, "Angle_test_final")
    #     x1 = points[0][0]
    #     y1 = points[0][1]
    #     x2 = points[1][0]
    #     y2 = points[1][1]
    #     seen_pixel_target_line = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     alpha = RotateAngle_z(focal_length=focal_length, real_target_line_length=KNOWN_WIDTH,
    #                           real_distance=real_distance, seen_pixel_target_line=seen_pixel_target_line)
    #     print("alpha is {}".format(alpha))
