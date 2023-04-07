import cv2
import numpy as np
from my_Camera import camera
import os


# 使用手机摄像头进行采集

def Hough_Detect(img, HoughLinesThreshold=160, if_show=0):
    img1 = img.copy()
    # img2 = img.copy()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_treshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    # cv2.imshow("gray", gray)
    # edges = cv2.Canny(binary, 200, 500, apertureSize=3)
    edges = cv2.Canny(binary, 120, 200)
    cv2.imshow("edge", edges)
    hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, HoughLinesThreshold)

    lines = []
    thick_flag = 0

    for hough_line in hough_lines:
        rho = hough_line[0][0]
        theta = hough_line[0][1]
        a =  float(np.cos(theta))
        b =  float(np.sin(theta))
        x0 = float(a * rho)
        y0 = float(b * rho)
        x1 = float(x0 + 1000 * (-b))
        y1 = float(y0 + 1000 * (a))
        x2 = float(x0 - 1000 * (-b))
        y2 = float(y0 - 1000 * (a))

        thick_flag = thick_flag + 2
        cv2.line(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thick_flag)
        lines.append([[x1, y1], [x2, y2]])

    # print(hough_lines)
    # print(":::::")
    # print(lines)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HoughLinesThreshold, 300, 5)
    #
    # for line in lines:
    #     x1 = line[0][0]
    #     y1 = line[0][1]
    #     x2 = line[0][2]
    #     y2 = line[0][3]
    #     cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    # cv2.imshow('edges', img2)

    if if_show == 1:
        cv2.imshow('houghlines3', img1)
        cv2.waitKey(0)

    return lines



def nothing(x):  # 滑动条的回调函数
    pass

def Hough_2(imgg):
    src = imgg
    srcBlur = cv2.GaussianBlur(src, (3, 3), 0)
    gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    WindowName = 'Approx'  # 窗口名
    cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口

    cv2.createTrackbar('threshold', WindowName, 0, 60, nothing)  # 创建滑动条

    while (1):
        img = src.copy()
        threshold = 100 + 2 * cv2.getTrackbarPos('threshold', WindowName)  # 获取滑动条值

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

        thick_flag = 1
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            thick_flag = thick_flag + 20
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=thick_flag)

        cv2.imshow(WindowName, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

def CrossPoint(line1, line2):  # 计算交点函数
    #是否存在交点
    point_is_exist = 0
    x = 0
    y = 0
    x1 = line1[0][0]  # 取四点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]

    if (x2 - x1) == 0:
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        # k1 = (y2 - y1) / (x2 - x1)
        # b1 = y1 - x1 * k1

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if k2 is not None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = 1
    elif k2 is None:
        x = x3
        y = k1 * x3+b1
        point_is_exist = 1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = 1
    return point_is_exist, [x, y]

def PerspectiveAngleCalculate(line1,line2):
    x1 = line1[0][0]
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]

    arr_0 = np.array([(x2 - x1), (y2 - y1)])
    arr_1 = np.array([(x4 - x3), (y4 - y3)])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    return np.arccos(cos_value) * (180 / np.pi)
    # return np.arccos(cos_value)

def my_main(img, initial_hough_threshold=160, if_show=0, if_debug=1):

    # release 版本打开下面的注释
    if if_debug == 0:
        Hough_Detect(img, if_show=1)
        while(1):
            if_continue_hough = input("add hough:1; decrease hough:-1; break hough:0;")
            if_continue_hough = int(if_continue_hough)
            if if_continue_hough == 0:
                final_hough_threshold = initial_hough_threshold
                break
            elif if_continue_hough == 1:
                initial_hough_threshold = initial_hough_threshold + 5
                Hough_Detect(img, HoughLinesThreshold=initial_hough_threshold, if_show=1)
            elif if_continue_hough == -1:
                initial_hough_threshold = initial_hough_threshold - 5
                Hough_Detect(img, HoughLinesThreshold=initial_hough_threshold, if_show=1)
        print("here is break")
        lines = Hough_Detect(img, HoughLinesThreshold=final_hough_threshold, if_show=1)

    elif if_debug == 1:
        lines = Hough_Detect(img, HoughLinesThreshold=160, if_show=1)

    #下一步求出透视角交点

    # 判断沿着z轴旋转的角度

    # 交点
    cross_points = []
    # cross_points:
    #   if_exist_cross_point, 点是否存在
    #   [cross_points_x, cross_points_y], 点坐标
    #   [i, j] 点由lines中的第几个线段相交而成

    for i in range(len(lines) - 1):
        for j in range(i+1, len(lines)):
            if_exist_cross_point, [cross_points_x, cross_points_y] = CrossPoint(lines[i], lines[j])
            if if_exist_cross_point != 0:
                cross_points.append([if_exist_cross_point, [cross_points_x, cross_points_y], [i, j]])
            else:
                print("lines{} and lines{} do not have a cross point".format(i, j))
    # print("cross_points are:")
    # print(cross_points)

    # 取最右边的交点为绕着z轴旋转产生的透视角，注意，这里的透视角是全角度，不是半角度

    xs = []
    for cross_point in cross_points:
        xs.append(cross_point[1][0])
    xs_max_index = xs.index(max(xs))
    print("右边透视点坐标为：")
    print(cross_points[xs_max_index][1])
    line1 = lines[cross_points[xs_max_index][2][0]]
    line2 = lines[cross_points[xs_max_index][2][1]]

    perspective_angle = PerspectiveAngleCalculate(line1, line2)
    print("perspective_angle is {}".format(perspective_angle))


if __name__ == '__main__':
    img_name = '60c1.jpg'
    image_file = os.path.join(str(r'D:\my_Academic\teacher_zyb\anglefile\60c'), img_name)
    # camera(save_path=image_file, camera_kind=0)
    img = cv2.imread(image_file)
    img = cv2.resize(img, (720, 720))
    # cv2.getRotationMatrix2D(center=(360, 360), angle=180, scale=1)
    img = cv2.flip(img, 1)

    # image_harris = HarrisDetect(img)
    # lineDetection(img)
    # cv2.imshow('img_Harris', image_harris)
    # Hough_Detect(img)
    # Hough_2(img)

    my_main(img, initial_hough_threshold=140, if_debug=0)
