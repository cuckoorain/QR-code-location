from PaddleDetection import my_infer0722 as my_Model
import cv2 as cv
import os
import time
import PIL
import my_Camera
from PIL import Image
import my_Staged_Post_Processing as PostProcess
import find_square as fs
import cv2

def QR_detector(img_name,if_show = 0):

    model_dir = str(r'D:\my_Academic\teacher_zyb\PaddleDetection\output\inference_model\ppyolo_mbv3_large_qr')
    image_file_name = os.path.join(str(r'D:\my_Academic\teacher_zyb\testfile'), img_name)
    output_dir = str(r'D:\my_Academic\teacher_zyb\outputfile')
    output_dir_error = str(r'D:\my_Academic\teacher_zyb\outputfile_error')
    output_dir_val = str(r'D:\my_Academic\teacher_zyb\outputfile_val')


    # 拍照
    # my_Camera.camera(save_path=image_file_name, camera_kind=0)

    # 图像推理得出二维码是否存在以及其方位，最后一个参数为0是为了防止有框的图片放到output文件夹导致后面图像处理的时候最大框数量增大
    bbox = my_Model.my_Main(model_dir, image_file_name, output_dir, if_show=if_show)
    # 如果要查看标定框，打开下面一行的代码，并在output_dir_val文件夹中查看
    bbox = my_Model.my_Main(model_dir, image_file_name, output_dir_val, if_show=if_show)


    # xmin, ymin, xmax, ymax = bbox
    # print('xmin is {}'.format(xmin))
    # print('ymin is {}'.format(ymin))
    # print('xmax is {}'.format(xmax))
    # print('ymax is {}'.format(ymax))
    # # 现实图片用cv必须要运行两次程序才行，所以用PIL
    # img = cv.imread(os.path.join(output_dir, img_name), 1)
    # cv.namedWindow('IMG')
    # cv.imshow("IMG", img)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # 展示原始图像
    # img = Image.open(os.path.join(output_dir, img_name))
    # img.show()


    # 图像切割
    bbox_int = []
    for index in range(0, len(bbox)):
        bbox_int.append(int(bbox[index]))
    seg_img_name = PostProcess.image_seg(ori_img_name = img_name,
                                              ori_img_path=output_dir,
                                              seg_img_path=output_dir,
                                              bbox=bbox_int)

    # RGB2YUV
    # YUV_img = PostProcess.RGB2YUV(seg_img_path=output_dir,seg_img_name=seg_img_name)

    # RGB2Gray
    Gray_img_name, Gray_img = PostProcess.RGB2Gray(FormerProcess_img_path=output_dir, FormerProcess_img_name=seg_img_name)

    # Gray2HE_Gray
    # HE_Gray_img = PostProcess.histogram_equalization(FormerProcess_img_path=output_dir, FormerProcess_img_name=Gray_img_name)

    # Gray_Hist
    # PostProcess.GrayHist_Plot(os.path.join(output_dir, Gray_img_name))

    squares = fs.find_square(os.path.join(output_dir, Gray_img_name),if_show=if_show)

    # 异常值检测
    if squares == -1:
        # 这一段代码在测试的时候出现：UnboundLocalError: local variable 'bbox' referenced before assignment
        # 猜测可能是无法识别出二维码（因为就算是拉伸过后，二维码也太模糊了）
        # 或许只有高分辨率可以做到这一点
        #########################################
        # 将有检测框的放入output_dir_error文件夹中
        image_file_name_error = os.path.join(output_dir, str('seg_') + str(img_name))
        img = cv2.imread(image_file_name_error)
        # 统一放到720*720
        img = cv2.resize(img, (720, 720), cv2.INTER_LINEAR)
        image_file_name_error = os.path.join(output_dir, str('error_') + str('seg_') + str(img_name))
        cv2.imwrite(image_file_name_error, img)
        bbox = my_Model.my_Main(model_dir, image_file_name_error, output_dir_error, if_show=if_show)
        img = Image.open(os.path.join(output_dir_error, img_name))
        img.show()
    else:
        squares_points = [i[-1] for i in squares]
        print(squares_points)

if __name__ == '__main__':
    img_name = 'test230216.jpg'
    QR_detector(img_name=img_name)
