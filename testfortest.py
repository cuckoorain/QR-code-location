import cv2  # 导入opencv
import math  # 导入math库,计算角度时需要
import os

width = 708.7
length = 992.5

#单击显示坐标位置
def LeftClikcSHow(imglb):
    points = LeftClick(imglb)
    while (1):
        cv2.imshow("image", imglb)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return points

def LeftClick(imglb):
    points = []
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            points.append([x, y])
            cv2.circle(imglb, (x, y), 1, (255, 0, 0), thickness = -1)
            cv2.putText(imglb, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0,0,0), thickness = 1)
            cv2.imshow("image", imglb)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    return points

if __name__=="__main__":
    img_name = "50cm0.jpg"
    img_file = os.path.join(str(r'D:\my_Academic\teacher_zyb\test_focal'), img_name)
    img = cv2.imread(img_file)
    cv2.imshow("00", img)
    rows, cols, channel = img.shape
    print(img.shape)
    print(rows)
    print(cols)
    print(int(1/10*rows))
    img0 = img[0:int(1/2*rows), 0:int(1 / 2 * cols)]
    img1 = img[int(1 / 2 * rows):rows,0:int(1 / 2 * cols)]
    img2 = img[0:int(1 / 2 * rows),int(1 / 2 * cols):cols]
    img3 = img[int(1 / 2 * rows):cols,int(1 / 2 * cols):cols]
    # cv2.imshow("0", img0)
    # cv2.imshow("1", img1)
    # cv2.imshow("2", img2)
    # cv2.imshow("3", img3)
    # cv2.waitKey(0)

    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

    points = LeftClikcSHow(imglb=img)
    print(points)

