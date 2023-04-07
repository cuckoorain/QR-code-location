import numpy as np
import cv2
import os

#单击显示坐标位置
def LeftClikcShow(imglb, task_name):
    points = LeftClick(imglb, task_name)
    while (1):
        cv2.imshow(str(task_name), imglb)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return points

def LeftClick(imglb, task_name):
    points = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            points.append([x, y])
            cv2.circle(imglb, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(imglb, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            cv2.imshow(str(task_name), imglb)
    cv2.namedWindow(str(task_name))
    cv2.setMouseCallback(str(task_name), on_EVENT_LBUTTONDOWN)
    return points

def SortPointsFromClick(points):
    # 从左上到右下排布
    sumx = 0
    sumy = 0
    for [x, y] in points:
        sumx = sumx + x
        sumy = sumy + y
    avgx = 1.0 * sumx / len(points)
    avgy = 1.0 * sumy / len(points)
    sorted_points = [[0] * 2] * 4
    for point in points:
        if point[1] <= avgy:
            if point[0] <= avgx:
                sorted_points[0] = point
            elif point[0] > avgx:
                sorted_points[1] = point
        elif point[1] > avgy:
            if point[0] <= avgx:
                sorted_points[2] = point
            elif point[0] > avgx:
                sorted_points[3] = point
    return sorted_points


def FindMarkerByClick(img, task_name="DEFAULT"):
    points = LeftClikcShow(img, task_name)
    sorted_points = SortPointsFromClick(points)
    width = (np.sqrt((sorted_points[0][0] - sorted_points[1][0])**2 + (sorted_points[0][1] - sorted_points[1][1])**2) +
             np.sqrt((sorted_points[2][0] - sorted_points[3][0])**2 + (sorted_points[2][1] - sorted_points[3][1])**2)) / 2
    height = (np.sqrt((sorted_points[0][0] - sorted_points[2][0])**2 + (sorted_points[0][1] - sorted_points[2][1])**2) +
             np.sqrt((sorted_points[1][0] - sorted_points[3][0])**2 + (sorted_points[1][1] - sorted_points[3][1])**2)) / 2
    return [sorted_points, width, height]


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    image_gray = cv2.resize(gray, (1080, 720))
    cv2.imshow("image_gray", image_gray)

    # 二值化
    ret, imgThresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    image_Thresh = cv2.resize(imgThresh, (1080, 720))
    cv2.imshow("image_gray", image_Thresh)

    edged = cv2.Canny(imgThresh, 1, 3)  # Canny算子边缘检测

    # edged = cv2.Canny(gray, 35, 125)
    image_edged = cv2.resize(edged, (1080, 720))
    cv2.imshow("image_edged", image_edged)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 求最大面积
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the of the paper region and return it
    # cv2.minAreaRect() c代表点集，返回rect[0]是最小外接矩形中心点坐标，
    # rect[1][0]是width，rect[1][1]是height，rect[2]是角度
    return cv2.minAreaRect(c)


# 距离计算函数
def DistancetToCamera_Width(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

def DistancetToCamera_Height(knownHeighth, focalLength, perHeighth):
    # compute and return the distance from the maker to the camera
    return (knownHeighth * focalLength) / perHeighth

# 计算测试距离
def DistanceTest(main_path, test_distance_input, KNOWN_WIDTH, KNOW_HEIGHTH, focalLength, if_by_width = 1):
    img_name_test = str(test_distance_input) + "cm.jpg"
    img_file_test = os.path.join(main_path, img_name_test)
    img_test = cv2.imread(img_file_test)
    points_attributes_test = FindMarkerByClick(img_test, task_name="DistanceTest of " + img_name_test)
    print("markers of test are")
    print(points_attributes_test)
    if if_by_width == 1:
        distance = DistancetToCamera_Width(KNOWN_WIDTH, focalLength, points_attributes_test[1])
    elif if_by_width == 0:
        distance = DistancetToCamera_Height(knownHeighth= KNOW_HEIGHTH, perHeighth=points_attributes_test[2])
    else :
        distance = DistancetToCamera_Height(knownHeighth=KNOW_HEIGHTH, perHeighth=points_attributes_test[2])
    print("distance is {}".format(distance))
    return distance

# 计算纸张的主函数
def DistanceCalculate(main_path, KNOWN_DISTANCE = 50, KNOWN_WIDTH = 29.7, KNOWN_HEIGHT = 21):

    img_name = "50cm.jpg"
    img_file = os.path.join(main_path, img_name)
    img = cv2.imread(img_file)
    points_attributes = FindMarkerByClick(img, task_name = "FocalTest")
    focalLength = (points_attributes[1] * KNOWN_DISTANCE) / KNOWN_WIDTH
    print(focalLength)
    print("Focal Detect End!!!")

    while(1):
        if_continue_flag = input("if continue testing: 1 for yes 0 for no")
        if int(if_continue_flag) == 0:
            cv2.destroyAllWindows()
            break
        else:

            # 测试纸张到摄像头的长度
            test_distance_input = input("choose distance:20/50/100/200/1000")
            distance = DistanceTest(main_path, test_distance_input, KNOWN_WIDTH, KNOWN_HEIGHT, focalLength=focalLength , if_by_width=1)

            # #测试线段距离摄像头的长度
            # img_name_line = "30c.jpg"
            # img_file_line = os.path.join(main_path, img_name_line)
            # img_line = cv2.imread(img_file_line)
            # #测width所在线段
            # points_line = LeftClikcShow(img_line, "LineTest")
            # print("points_line are {}".format(points_line))
            # distance = DistanceTest_forLine(points_line, focal_length=focalLength, real_line_length=KNOWN_WIDTH)
            # cv2.waitKey(0)
    return distance

# 计算线段的主函数
def DistanceCalculate_forLine(main_path, KNOWN_DISTANCE = 50, KNOWN_WIDTH = 29.7, KNOWN_HEIGHT = 21,
                              if_width = 0, if_retest_focal = 0, focal_length_input = 0, if_return_x = 0):
    if if_retest_focal == 1:
        img_name = "50cm.jpg"
        img_file = os.path.join(main_path, img_name)
        img = cv2.imread(img_file)
        print("DistanceCalculate_forLine FocalTest shape is {}".format(img.shape))
        points_attributes = FindMarkerByClick(img, task_name = "FocalTest")
        focalLength = (points_attributes[1] * KNOWN_DISTANCE) / KNOWN_WIDTH
    elif if_retest_focal == 0:
        focalLength = focal_length_input
    print(focalLength)
    print("Focal Detect End!!!")

    while(1):
        if_continue_flag = input("if continue testing: 1 for yes 0 for no")
        if int(if_continue_flag) == 0:
            cv2.destroyAllWindows()
            break
        else:

            # # 测试纸张到摄像头的长度
            test_name_input = input("test_name_input e.g. 50cm")
            # distance = DistanceTest(main_path, test_distance_input, KNOWN_WIDTH, focalLength, if_by_width=1)
            test_name_input = test_name_input + ".jpg"
            #测试线段距离摄像头的长度
            img_name_line = test_name_input
            img_file_line = os.path.join(main_path, img_name_line)
            img_line = cv2.imread(img_file_line)
            print("DistanceCalculate_forLine  AngleTest shape is {}".format(img_line.shape))
            #测width所在线段
            points_line = LeftClikcShow(img_line, "AngleTest")
            print("points_line are {}".format(points_line))

            if if_return_x ==1:
                print("if_return_x is 1")
                return_x_pixel = 1.0 * (points_line[0][0] + points_line[1][0]) / 2 - img_line.shape[0]/2

            if if_width == 1:
                distance = DistanceTest_forLine(points_line, focal_length=focalLength, real_line_length=KNOWN_WIDTH)
            else:
                distance = DistanceTest_forLine(points_line, focal_length=focalLength, real_line_length=KNOWN_HEIGHT)
            cv2.waitKey(0)
    if if_return_x == 0:
        return distance, focalLength
    if if_return_x == 1:
        return distance, focalLength, return_x_pixel



# 计算线段到摄像头之间的距离
def DistanceTest_forLine(points, focal_length, real_line_length):
    """
    计算一条线段到摄像头的距离（假设线段所处平面与摄像头平面平行）
    :param points:两个点的[x, y]坐标
    :param focal_length:本次实验测得的焦距
    :param real_line_length:该线段的物理实际长度
    :return:本线段距离摄像头的显示长度（cm）
    """
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    pixel_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    real_distance = (real_line_length * focal_length) / pixel_length
    return real_distance



if __name__=="__main__":
    main_path = str(r'D:\my_Academic\teacher_zyb\test_distance')
    KNOWN_DISTANCE = 50

    # A4纸的长和宽(单位:cm)
    # 这个方法的鲁棒性不是很强，但是没关系，因为我不需要这个模块做图像处理，我只需要
    # 这个模块做相似三角形的计算而已，到时候传递进来的是坐标而不是其它的东西
    KNOWN_WIDTH = 29.7
    KNOWN_HEIGHT = 21
    # focalLength = 1108.7290463350978
    distance = DistanceCalculate_forLine(main_path=main_path)
    # distance = DistanceCalculate(main_path=main_path)

    print("calculate by line for distance is {}".format(distance))






