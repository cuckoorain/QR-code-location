from my_Camera import camera
import os
from my_QR_scanner import QR_detector
from find_square import find_square
from my_QR_Decoder import QRcode_decoder
from SpaceAngle_PerspectiveAngle import Angles_Function
import cv2


def my_Main(img_name, if_debug=0, camera_kind=0, if_show_detector=0, if_show_find_square=0, if_show_find_square_squares=0,
            if_show_decoder=0):
    """
    :param img_name: 以包括.jpg格式的形式输入图片名称
    :param if_debug: 0为Release版本，1为Debug版本
    :param camera_kind: 0为默认相机，需要运行后调试
    :param if_show_detector: 是否展示QR_detector的输出结果
    :param if_show_find_square: 是否展示find_square的图片输出结果
    :param if_show_find_square_squares: 是否展示find_square中三个定位点的详细信息
    :param if_show_decoder: 是否展示QRcode_decoder的输出结果
    :return:
    """
    # 拍照
    print("My_Main.拍照 Start!")
    if if_debug == 0:
        image_file = os.path.join(str(r'D:\my_Academic\teacher_zyb\testfile'), img_name)
        camera(save_path=image_file, camera_kind=camera_kind)
    elif if_debug == 1:
        image_file = str(r'D:\my_Academic\teacher_zyb\testfile\test230216.jpg')
    print("My_Main.拍照 End!")

    # 二维码识别
    print("My_Main.二维码识别 Start!")
    QR_detector(img_name=img_name, if_show=if_show_detector)
    print("My_Main.二维码识别 End!")

    # 二维码找定位点
    print("My_Main.二维码找定位点 Start!")
    path_name = './outputfile/seg_' + str(img_name)
    squares = find_square(path_name, if_show=if_show_find_square)
    squares_points = [i[-1][0][3] for i in squares]
    if if_show_find_square_squares == 1:
        print(squares_points)
        print("squares are:\n")
        print(squares)
    print("My_Main.二维码找定位点 End!")


    # 二维码解码
    print("My_Main.二维码解码 Start!")
    image = cv2.imread(image_file)
    QRcode_decoder(image, if_show=if_show_decoder)
    print("My_Main.二维码解码 End!")


    # # 空间角与透视角函数关系测试
    # print("My_Main.空间角与透视角函数关系测试 Start!")
    # Angles_Function(squares)
    # print("My_Main.空间角与透视角函数关系测试 End!")


if __name__ == '__main__':
    img_name = 'test230216.jpg'
    my_Main(img_name, if_debug=1, if_show_find_square_squares=1)
