import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def image_seg(ori_img_name,ori_img_path,seg_img_path, bbox, if_show=0, if_dialte=1):
    xmin, ymin, xmax, ymax = bbox
    seg_img_name = 'seg_' + str(ori_img_name)
    img = cv2.imread(os.path.join(ori_img_path,ori_img_name))
    print(img.shape)

    if if_dialte == 0:
        cut_ymin = ymin
        cut_ymax = ymax
        cut_xmin = xmin
        cut_xmax = xmax
    else:
        w = xmax - xmin
        h = ymax - ymin
        rate = 1/3
        cut_ymin = int(ymin - rate * h)
        cut_ymax = int(ymax + rate * h)
        cut_xmin = int(xmin - rate * w)
        cut_xmax = int(xmax + rate * w)


    img = img[cut_ymin:cut_ymax, cut_xmin:cut_xmax]  # 裁剪坐标为[y0:y1, x0:x1]
    seg_img_path_name = os.path.join(seg_img_path, seg_img_name)
    cv2.imwrite(seg_img_path_name, img)
    if if_show == 0:
        pass
    else:
        cv2.imshow("seg_img",img)
    return str(seg_img_name)

def RGB2YUV(seg_img_path,seg_img_name, YUV_img_path = None, if_show=0):
    # 命名
    YUV_img_name = 'YUV_' + str(seg_img_name)
    if YUV_img_path is None:
        YUV_img_path = seg_img_path
    elif YUV_img_path is None:
        pass
    seg_img_path_name = os.path.join(seg_img_path, seg_img_name)
    YUV_img_path_name = os.path.join(YUV_img_path, YUV_img_name)
    # 转YUV
    RGB_img = cv2.imread(str(seg_img_path_name))
    YUV_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2YUV)
    # 输出形式
    if if_show == 0:
        cv2.imwrite(YUV_img_path_name, YUV_img)
    elif if_show == 1:
        cv2.imwrite(YUV_img_path_name, YUV_img)
        return YUV_img_name, YUV_img

def RGB2Gray(FormerProcess_img_path,FormerProcess_img_name, Gray_img_path = None, if_show=0):
    # 命名
    Gray_img_name = 'Gray_' + str(FormerProcess_img_name)
    if Gray_img_path is None:
        Gray_img_path = FormerProcess_img_path
    elif Gray_img_path is None:
        pass
    FormerProcess_img_path_name = os.path.join(FormerProcess_img_path, FormerProcess_img_name)
    Gray_img_path_name = os.path.join(Gray_img_path, Gray_img_name)
    # 转Gray
    RGB_img = cv2.imread(str(FormerProcess_img_path_name))
    Gray_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)
    # 输出形式
    if if_show == 0:
        cv2.imwrite(Gray_img_path_name, Gray_img)
        return Gray_img_name, Gray_img
    elif if_show == 1:
        cv2.imwrite(Gray_img_path_name, Gray_img)
        Gray_img.show()
        return Gray_img_name, Gray_img

def histogram_equalization(FormerProcess_img_path, FormerProcess_img_name, HE_img_path=None, if_show=0):
    # 命名
    Gray_img_name = 'HE_' + str(FormerProcess_img_name)
    if HE_img_path is None:
        HE_img_path = FormerProcess_img_path
    elif HE_img_path is None:
        pass
    FormerProcess_img_path_name = os.path.join(FormerProcess_img_path, FormerProcess_img_name)
    HE_img_path_name = os.path.join(HE_img_path, Gray_img_name)
    # HE增强
    RGB_img = cv2.imread(str(FormerProcess_img_path_name))
    img = cv2.imread(FormerProcess_img_path_name, 1)  # 此图片是彩色图，名为caise2.jpg
    (b, g, r) = cv2.split(img)  # 通道分解
    bH = cv2.equalizeHist(b)  # 函数equalizeHist用来做直方图均衡化
    gH = cv2.equalizeHist(g)  # 函数equalizeHist只支持单通道的灰度图
    rH = cv2.equalizeHist(r)  # 彩色图要多通道分离成单通道，然后再合并成多通道
    result = cv2.merge((bH, gH, rH), )  # 通道合成
    res = np.hstack((img, result))
    cv2.imshow("比较",res)
    cv2.waitKey(0)
    # 输出形式
    if if_show == 0:
        cv2.imwrite(HE_img_path_name, img)
    elif if_show == 1:
        cv2.imwrite(HE_img_path_name, img)
        return img

# 灰度直方图
def calcGrayHist(image):
    #灰度图像矩阵的高、宽
    rows, cols = image.shape
    #存储灰度直方图
    grayHist=np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] +=1
    return grayHist

# 绘制灰度直方图
def GrayHist_Plot(img_path_name):
    #第一个参数式图片地址，你只需放上你的图片就可
    image = cv2.imread(img_path_name, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("image", image)
    print("Usge:python histogram.py imageFile")
    #计算灰度直方图
    grayHist=calcGrayHist(image)
    #画出灰度直方图
    x_range=range(256)
    plt.plot(x_range,grayHist,'r',linewidth=2,c='black')
    #设置坐标轴的范围
    y_maxValue=np.max(grayHist)
    plt.axis([0,255,0,y_maxValue])
    plt.ylabel('gray level')
    plt.ylabel("number or pixels")
    # 显示灰度直方图
    plt.show()


# 图像轻度膨胀
def img_dialte():
    1/2


if __name__ == '__main__':
    img_name = 'test1031.jpg'
    model_dir = str(r'D:\my_Academic\teacher_zyb\PaddleDetection\output\inference_model\ppyolo_mbv3_large_qr')
    image_file = os.path.join(str(r'D:\my_Academic\teacher_zyb\testfile'), img_name)
    output_dir = str(r'D:\my_Academic\teacher_zyb\outputfile')
    image_seg(ori_img_name = img_name, ori_img_path=output_dir,seg_img_path=output_dir,bbox=[0,0,120,124],if_show=1)


    # img = cv2.imread('caise2.jpg', 1)  # 此图片是彩色图，名为caise2.jpg
    # (b, g, r) = cv2.split(img)  # 通道分解
    # bH = cv2.equalizeHist(b)  # 函数equalizeHist用来做直方图均衡化
    # gH = cv2.equalizeHist(g)  # 函数equalizeHist只支持单通道的灰度图
    # rH = cv2.equalizeHist(r)  # 彩色图要多通道分离成单通道，然后再合并成多通道
    # result = cv2.merge((bH, gH, rH), )  # 通道合成
    # res = np.hstack((img, result))
    # # cv2.imshow('dst',res)
    # cv2.imshow('caise1', res)
    # cv2.waitKey(0)

