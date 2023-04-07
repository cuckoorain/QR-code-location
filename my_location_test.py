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
def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


if __name__=="__main__":


    KNOWN_DISTANCE = 50

    # A4纸的长和宽(单位:cm)
    # 这个方法的鲁棒性不是很强，但是没关系，因为我不需要这个模块做图像处理，我只需要
    # 这个模块做相似三角形的计算而已，到时候传递进来的是坐标而不是其它的东西
    KNOWN_WIDTH = 29.7
    KNOWN_HEIGHT = 21
    # focalLength = 1108.7290463350978
    img_name = "50cm0.jpg"
    img_file = os.path.join(str(r'D:\my_Academic\teacher_zyb\test_focal'), img_name)
    img = cv2.imread(img_file)
    points_attributes = FindMarkerByClick(img, task_name = "FocalTest")
    focalLength = (points_attributes[1] * KNOWN_DISTANCE) / KNOWN_WIDTH
    print(focalLength)
    print("Focal Detect End!!!")

    img_name_test = "50cm1.jpg"
    img_file_test = os.path.join(str(r'D:\my_Academic\teacher_zyb\test_focal'), img_name_test)
    img_test = cv2.imread(img_file_test)
    points_attributes_test = FindMarkerByClick(img_test, task_name="DistanceTest")
    print("markers of test are")
    print(points_attributes_test)
    distance = distance_to_camera(KNOWN_WIDTH, focalLength, points_attributes_test[1])
    # image = cv2.resize(img_test, (1080, 720))
    # draw a bounding box around the image and display it
    # box = np.int0(cv2.boxPoints(points_attributes_test[0]))

    print("distance is {}".format(distance))
    # # cv2.drawContours(img_test, [box], -1, (0, 255, 0), 3)
    #
    # # inches 转换为 cm
    # cv2.putText(img_test, "%.2fcm" % (distances),
    #             (img_test.shape[1] - 200, img_test.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    #             2.0, (0, 255, 0), 3)
    #
    # # show a frame
    #
    # cv2.imshow("capture", img_test)

cv2.waitKey(0)
cv2.destroyAllWindows()


    #
    # # initialize the list of images that we'll be using
    # IMAGE_PATHS = [str(r'D:\my_Academic\teacher_zyb\camera_parameters\picture1.jpg'),
    #                str(r'D:\my_Academic\teacher_zyb\camera_parameters\picture2.jpg'),
    #                str(r'D:\my_Academic\teacher_zyb\camera_parameters\picture3.jpg'),
    #                str(r'D:\my_Academic\teacher_zyb\camera_parameters\picture4.jpg'),
    #                str(r'D:\my_Academic\teacher_zyb\camera_parameters\picture5.jpg'),
    #                str(r'D:\my_Academic\teacher_zyb\camera_parameters\picture6.jpg')
    #                ]
    # # 读入第一张图，通过已知距离计算相机焦距
    # image = cv2.imread(IMAGE_PATHS[0])
    # marker = find_marker(image)
    # focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    #
    # # 通过摄像头标定获取的像素焦距
    # # focalLength = 811.82
    # print('focalLength = ', focalLength)
    #
    # # 控制是否打开摄像头还是判定已有照片
    # if_open_camera = 0
    #
    # if if_open_camera == 1:
    #     # 打开摄像头
    #     camera = cv2.VideoCapture(1)
    #
    #     while camera.isOpened():
    #         # get a frame
    #         (grabbed, frame) = camera.read()
    #         marker = find_marker(frame)
    #         if marker == 0:
    #             print(marker)
    #             continue
    #         inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    #
    #         # draw a bounding box around the image and display it
    #         box = np.int0(cv2.boxPoints(marker))
    #         cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    #
    #         # inches 转换为 cm
    #         cv2.putText(frame, "%.2fcm" % (inches * 30.48 / 12),
    #                     (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    #                     2.0, (0, 255, 0), 3)
    #
    #         # show a frame
    #         cv2.imshow("capture", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     camera.release()
    #
    # else:
    #     image = cv2.imread(IMAGE_PATHS[4])
    #     marker = find_marker(image)
    #     print("markers are" + str(marker))
    #     distances = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    #     image = cv2.resize(image, (1080, 720))
    #     # draw a bounding box around the image and display it
    #     box = np.int0(cv2.boxPoints(marker))
    #     cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    #
    #     # inches 转换为 cm
    #     cv2.putText(image, "%.2fcm" % (distances),
    #                 (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    #                 2.0, (0, 255, 0), 3)
    #
    #     # show a frame
    #
    #     cv2.imshow("capture", image)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
