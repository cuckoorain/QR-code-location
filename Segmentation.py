import cv2

import numpy as np

# 使用摄像头
cap = cv2.VideoCapture(1)
if __name__ == '__main__':
    # 循环
    while (1):

        # 从摄像头读取帧
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        # 显示
        cv2.imshow('Original', frame)

        # canny方法

        edges = cv2.Canny(frame, 100, 200)

        cv2.imshow('Edges', edges)

        # 按ESC暂停
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    # 关闭
    cap.release()

    cv2.destroyAllWindows()
