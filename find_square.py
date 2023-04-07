import  sys
import cv2
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import my_Staged_Post_Processing as PostProcess

#对于8位图，图像的灰度级范围式0~255之间的整数，通过定义函数来计算直方图
def calcGrayHist(image):
    #灰度图像矩阵的高、宽
    rows, cols = image.shape
    #存储灰度直方图
    grayHist=np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] +=1
    return grayHist

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
    cv2.waitKeyEx(0)

import cv2
import numpy as np

#定义形状检测函数
def ShapeDetection(img_usedto_detect, img_usedto_plot,crooped_bias = 0, used_croop = -1):
    # used_croop = -1 默认不使用分块编码，0代表是左上角，1代表是右上角，2代表是左下角
    # crooped_bias = 0默认对x和y不进行操作



    contours,hierarchy = cv2.findContours(img_usedto_detect,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    alternative_counters = []
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        cv2.drawContours(img_usedto_plot, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv2.arcLength(obj, True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj, 0.05*perimeter, True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度(左下角)
        if crooped_bias == 1:
            # 对应右上角的定位点情况
            x = x + 360
            pass
        if crooped_bias == 2:
            # 对应左下角的定位点情况
            y = y + 360
            pass

        #轮廓对象分类
        if CornerNum ==4:
            objType ="R"
            if used_croop !=-1:
                alternative_counters.append([area, perimeter, CornerNum, [x, y, w, h], approx])
            else:
                alternative_counters.append([area, perimeter, CornerNum, [x, y, w, h], approx, used_croop])
            cv2.rectangle(img_usedto_plot,(x,y),(x+w,y+h),(0,0,255),2)  #绘制边界框
            cv2.putText(img_usedto_plot,objType,(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1)  #绘制文字
        else:
            pass

    return alternative_counters

# 画先看area的分布情况
def area_cruve(alternative_counters, if_show=0):
    area_alternative_counter = sorted(alternative_counters, key=lambda x: x[0])
    area_alternative_counter = np.asarray(area_alternative_counter)
    sorted_areas = area_alternative_counter[:, 0]
    d_areas = [ sorted_areas[i+1] - sorted_areas[i] for i in range(0,len(sorted_areas) -1)]

    if if_show == 1:
        plt.plot(sorted_areas, 'o')
        plt.show()
    else:
        pass


    return d_areas


def area_cruve_method3(alternative_counters, if_show=0):
    area_alternative_counter = sorted(alternative_counters, key=lambda x: x[0])
    area_alternative_counter = np.asarray(area_alternative_counter)
    sorted_areas = area_alternative_counter[:, 0]
    d_areas = [ sorted_areas[i+1] - sorted_areas[i] for i in range(0,len(sorted_areas) -1)]

    if if_show == 1:
        plt.plot(sorted_areas, 'o')
        plt.show()
    else:
        pass


    return d_areas


# 找出定位点
# 不用这个了
# def Position_Sqaure_detector(alternative_counters,img):
#     area_alternative_counter = sorted(alternative_counters, key=lambda x: x[0])
#     areas = [i[0] for i in area_alternative_counter]
#     area_avg = sum(areas)/len(areas)
#     effective_counter = []
#     for square_properties in area_alternative_counter:
#         [x,y,w,h] = square_properties[3]
#         # 接近正方形
#         # 重要的判断语句
#         # 排除掉最大的整体QR Code边缘以及内部的小正方体
#         if square_properties[0] > area_avg and square_properties[0] < 1 / 2 * area_alternative_counter[-1][0]:
#         # if square_properties[0] > area_alternative_counter[len(area_alternative_counter)//2][0] and square_properties[0] < 1 / 2 * area_alternative_counter[-1][0]:
#         # if 0.6 < w/h and w/h < 1/0.6 and square_properties[0] > area_avg and square_properties[0] < 1 / 4 * np.size(img):
#             effective_counter.append(square_properties)
#             cv2.rectangle(img_Square_detector, (x, y), (x + w, y + h), (127, 0, 255), 2)  # 绘制边界框

    # return effective_counter

# # 根据颜色判断在大定位点的哪一层
# def color_detector(effective_counter,img):




# 找出定位点方法2
def Position_Sqaure_detector_method2(alternative_counters,img_usedto_plot, max_area_begin_index, imgThresh):

    del alternative_counters[max_area_begin_index:-1]

    area_alternative_counter = sorted(alternative_counters, key=lambda x: x[0])
    areas = [i[0] for i in area_alternative_counter]
    area_avg = sum(areas)/len(areas)
    # img = img[ymin:ymax, xmin:xmax]  # 裁剪坐标为[y0:y1, x0:x1]

    effective_counter = []
    for square_properties in area_alternative_counter:
        [x,y,w,h] = square_properties[3]
        # 接近正方形
        # 重要的判断语句
        # 排除掉最大的整体二维码边缘以及内部的小正方体
        if square_properties[0] > area_avg and square_properties[0] < 1 / 2 * area_alternative_counter[-1][0]:

        # if square_properties[0] > area_avg and square_properties[0] < 1 / 2 * area_alternative_counter[-1][0]:

        # if 0.6 < w/h and w/h < 1/0.6 and square_properties[0] > area_avg and square_properties[0] < 1 / 4 * np.size(img):
            effective_counter.append(square_properties)
            # cv2.rectangle(img_usedto_plot, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边界框

    # 可以用二阶差分做，但我选择试试kmeans，主要是玩一下
    # 不行，效果不好，因为固定了类数为3，所以如果上一步筛选到了白色框，按这里没办法排除
    # effective_areas = [i[0] for i in effective_counter]
    # effective_areas = np.asarray(effective_areas)
    # x = effective_areas.reshape(-1,1)
    #
    # kmeans = KMeans(n_clusters=3).fit(x)
    # print('kmeans.labels_            is{}'.format(kmeans.labels_))
    # print('kmeans.cluster_centers_   is{}'.format(kmeans.cluster_centers_))
    #
    # temp_effective_counter = []
    # effective_areas = effective_areas.tolist()
    # for label in [0, 1, 2]:
    #     for index in range(0, len(x)):
    #         if kmeans.labels_[index] == label:
    #             temp_effective_counter.append(effective_counter[index])
    #             break

    temp_effective_counter = []
    effective_areas = [i[0] for i in effective_counter]
    effective_areas.insert(0,0)

    # 没有识别出来的，可能是因为模糊，这个时候跳转到粗略判断，只是返回bbox的四个点
    if len(effective_areas) == 1:
        print("-------  DetectionError  -----------")
        print('use rough_position because cannot recognize positioning points')
        print("------------------------------------")
        # 返回-1作为异常值检测
        error_value = -1
        return error_value

    for index in range(len(effective_areas) - 1):

        if abs(effective_areas[index + 1] - effective_areas[index]) < 1/1000 * effective_areas[index]:
            # 由于数据出现孪生，所以去除孪生中的一个值
            pass
        else:
            temp_effective_counter.append(effective_counter[index])

    collapsed_effective_counter = []
    for index in range(3):
        collapsed_effective_counter.append(max(temp_effective_counter))
        temp_effective_counter.pop()
        # 没有识别出来的，可能是因为模糊，这个时候跳转到粗略判断，只是返回bbox的四个点
        # if len(temp_effective_counter) == 0:
        #     print("-------  DetectionError  -----------")
        #     print('use rough_position because cannot recognize positioning points')
        #     print('in line 174')
        #     print("------------------------------------")
        #     # 返回-1作为异常值检测
        #     error_value = -1
        #     return error_value

    if len(collapsed_effective_counter) != 3:
        print("-------  DetectionError  -----------")
        print('use rough_position because cannot recognize positioning points')
        print('in line 184')
        print("------------------------------------")
        # 返回-1作为异常值检测
        error_value = -1
        return error_value

    for square_properties in collapsed_effective_counter:
        [x,y,w,h] = square_properties[3]
        cv2.rectangle(img_usedto_plot, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边界框

    return collapsed_effective_counter

# 找出定位点方法3
def Position_Sqaure_detector_method3(alternative_counters,img_usedto_plot, if_return_alternatives = 0):


    area_alternative_counter = sorted(alternative_counters, key=lambda x: x[0])
    areas = [i[0] for i in area_alternative_counter]
    area_avg = sum(areas)/len(areas)
    # img = img[ymin:ymax, xmin:xmax]  # 裁剪坐标为[y0:y1, x0:x1]

    effective_counter = []
    for square_properties in area_alternative_counter:
        [x,y,w,h] = square_properties[3]
        # 接近正方形
        # 重要的判断语句
        # 排除掉最大的整体二维码边缘以及内部的小正方体
        # if square_properties[0] > area_avg and square_properties[0] < 1 / 2 * area_alternative_counter[-1][0]:
        temp_effective_counter = []
        if w/h > 0.8 and h/w > 0.8 :
            temp_effective_counter.append(square_properties)

    effective_counter.append(max(temp_effective_counter))

    # 绘图
    for square_properties in effective_counter:
        [x,y,w,h] = square_properties[3]
        cv2.rectangle(img_usedto_plot, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边界框

    if if_return_alternatives ==0:
        return effective_counter

    # 判断出现失误，返回备选项
    else:
        return effective_counter, temp_effective_counter


# 主函数
def find_square(path_name, if_show = 0, method = 3):
    # image = cv2.imread('./outputfile/Gray_seg_test3.jpg', cv2.IMREAD_GRAYSCALE)
    # GrayHist_Plot('./outputfile/Gray_seg_test3.jpg')

    img = cv2.imread(path_name)

    # 统一放到720*720
    img = cv2.resize(img, (720, 720), cv2.INTER_LINEAR)


    # 使用方法3，做三个备份文件
    img_alternative_counters = []
    img_effective_counter = img.copy()
    img_alternative_counters.append(img.copy())
    img_alternative_counters.append(img.copy()[0:360    ,   0:360])
    img_alternative_counters.append(img.copy()[0:360    , 360:720])
    img_alternative_counters.append(img.copy()[360:720  ,   0:360])
    # 下面一行不写的原因是二维码右下角没有大方块（定位点）
    #img_alternative_counters.append(img.copy()[360:720, 360:720])


    img_ori = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图
    # 二值化
    ret, imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    # imgCanny = cv2.Canny(imgBlur, 60, 60)  # Canny算子边缘检测
    imgCanny = cv2.Canny(imgThresh, 60, 60)  # Canny算子边缘检测
    # alternative_counters = ShapeDetection(img_usedto_detect=imgCanny,img_usedto_plot=img_alternative_counters)  # 形状检测
    alternative_counters = ShapeDetection(img_usedto_detect=imgCanny,img_usedto_plot=img_alternative_counters[0])  # 形状检测
    # # 图像再裁剪
    # area_alternative_counter = sorted(alternative_counters, key=lambda x: x[0])
    # max_square = area_alternative_counter[-1]
    # x,y,w,h = max_square[3]
    # img_seg = img_ori[y:y+h, x:x+w]  # 裁剪坐标为[y0:y1, x0:x1]
    # img_alternative_counters_seg = img_ori.copy()
    # imgGray_seg = cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY)  # 转灰度图
    # # 二值化
    # ret_seg, imgThresh_seg = cv2.threshold(imgGray_seg, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #
    # # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    # # imgCanny = cv2.Canny(imgBlur, 60, 60)  # Canny算子边缘检测
    # imgCanny_seg = cv2.Canny(imgThresh_seg, 60, 60)  # Canny算子边缘检测
    # alternative_counters = ShapeDetection(img_usedto_detect=imgCanny_seg, img_usedto_plot=img_alternative_counters_seg)  # 形状检测




    areas_d = area_cruve(alternative_counters, if_show=if_show)


    if method == 2:
        # 还没测试过能不能跑通，不过函数本身没问题，可能就是数据格式有点问题
        effective_counter = Position_Sqaure_detector_method2(alternative_counters, img_effective_counter, 1 + areas_d.index(max(areas_d)), imgThresh)
        #异常值检测
        if effective_counter == -1:
            return -1

        # 绘制area_cruve
        area_cruve(effective_counter, if_show=if_show)

    if method == 3:
        # 另一种方法，用来解决定位块不是最大的黑色方形环的问题
        # 切分成四块
        crooped_imgThresh = []
        crooped_imgThresh.append(imgThresh[0:360    ,   0:360])
        crooped_imgThresh.append(imgThresh[0:360    , 360:720])
        crooped_imgThresh.append(imgThresh[360:720  ,   0:360])

        effective_counter = []
        alternative_counters = []
        imgCanny = []

        for crooped_index in range(3):
            # Canny算子边缘检测
            imgCanny.append(cv2.Canny(crooped_imgThresh[crooped_index], 60, 60))

            if crooped_index == 0:
                # 形状检测
                alternative_counters.append(ShapeDetection(img_usedto_detect=imgCanny[crooped_index],
                                                           img_usedto_plot=img_alternative_counters[crooped_index],
                                                           crooped_bias = 0,
                                                           used_croop = 0))
                # pass
            elif crooped_index == 1:
                # 形状检测
                alternative_counters.append(ShapeDetection(img_usedto_detect=imgCanny[crooped_index],
                                                           img_usedto_plot=img_alternative_counters[crooped_index],
                                                           crooped_bias = 1,
                                                           used_croop = 1))
            elif crooped_index == 2:
                # 形状检测
                alternative_counters.append(ShapeDetection(img_usedto_detect=imgCanny[crooped_index],
                                                           img_usedto_plot=img_alternative_counters[crooped_index],
                                                           crooped_bias = 2,
                                                           used_croop = 2))


            effective_counter.append(Position_Sqaure_detector_method3(alternative_counters[crooped_index],
                                                                      img_effective_counter,
                                                                      1 + areas_d.index(max(areas_d))
                                                                      ))

            # method3的area_cruve：
            area_cruve_method3(alternative_counters[crooped_index], if_show=if_show)

            # method3的if_show
            if if_show == 1:
                # cv2.imshow("imgCanny", imgCanny)
                cv2.imshow("imgCanny", imgCanny[crooped_index])
                cv2.imshow("shape Detection", img_alternative_counters[crooped_index + 1])
                cv2.imshow("img_Square_detector", img_effective_counter)
                cv2.waitKey(0)
            # else:
            #     pass
    return effective_counter







    # cv2.imshow("Original img", img)
    # cv2.imshow("imgGray", imgGray)
    # cv2.imshow("imgGray", imgThresh)
    # cv2.imshow("imgBlur", imgBlur)










if __name__=="__main__":
    path_name = './outputfile/seg_test230216.jpg'
    # 找到三个定位点的函数
    squares = find_square(path_name, if_show=1)

    # # image = cv2.imread('./outputfile/Gray_seg_test3.jpg', cv2.IMREAD_GRAYSCALE)
    # # GrayHist_Plot('./outputfile/Gray_seg_test3.jpg')
    # path = './outputfile/test1_seg.jpg'
    # img = cv2.imread(path)
    #
    # # 统一放到720*720
    # img = cv2.resize(img, (720, 720), cv2.INTER_LINEAR)
    # imgContour = img.copy()
    # img_Square_detector = img.copy()
    #
    # imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转灰度图
    # # 二值化
    # ret, imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #
    # # imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
    # # imgCanny = cv2.Canny(imgBlur, 60, 60)  # Canny算子边缘检测
    # imgCanny = cv2.Canny(imgThresh, 60, 60)  # Canny算子边缘检测
    # alternative_counters = ShapeDetection(imgCanny)  # 形状检测
    # areas_d = area_cruve(alternative_counters)
    #
    # effective_counter = Position_Sqaure_detector_method2(alternative_counters, img_Square_detector, 1 + areas_d.index(max(areas_d)))
    # # effective_counter = Position_Sqaure_detector(alternative_counters, img_Square_detector)
    # area_cruve(effective_counter)
    #
    # # cv2.imshow("Original img", img)
    # # cv2.imshow("imgGray", imgGray)
    # # cv2.imshow("imgGray", imgThresh)
    # # cv2.imshow("imgBlur", imgBlur)
    # cv2.imshow("imgCanny", imgCanny)
    # cv2.imshow("shape Detection", imgContour)
    # cv2.imshow("img_Square_detector", img_Square_detector)
    #
    # cv2.waitKey(0)


    # 检查
    # 目前还没有出现在这个需求，出现 的时候可以考虑用两种method进行验证

    # 定位
    # 三个大定位点的[x,y,w,h]
    squares_points = [i[-1][0][3] for i in squares]

    print(squares_points)
    print("squares are:\n")
    print(squares)
    # print(squares[0][0][0][4])