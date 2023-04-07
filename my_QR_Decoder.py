import cv2
import numpy as np
from pyzbar.pyzbar import decode    # 安装二维码解码包  pip install pyzbar

def QRcode_decoder(src_img,if_show):
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    barcode_objects = decode(src_img)
    if len(barcode_objects) == 0:
        print("!!!!!!Can not decode this QR code!!!!!!")

    elif len(barcode_objects) > 0:
        for obj in barcode_objects:
            print(obj)

            for j in range(0, 4):
                cv2.line(src_img, obj.polygon[j], obj.polygon[(j + 1) % 4], (255, 0, 0), 3)

        cv2.putText(src_img, barcode_objects[0][0].decode("utf-8"), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (125, 255, 125))
        print("this QR code is " + barcode_objects[0][0].decode("utf-8"))
    if if_show ==1:
        cv2.imshow('img', src_img)
        cv2.waitKey(0)

if __name__=="__main__":
    path_name = './outputfile/seg_test1031.jpg'
    # path_name = './outputfile/seg_test1031.jpg'
    image = cv2.imread(path_name)
    QRcode_decoder(image, if_show=1)
