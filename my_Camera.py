import cv2
import time
import os
auto_screenshoot = 1
# todo
# 这里的倒计时拍照不能实时预览，有空可以用多线程或是什么办法整一个

def camera(save_path, camera_kind=1):
    """
    :input 按键:a表示自动拍照；m表示手动拍照；k为退出进程
    :param save_path: 存储路径
    :param camera_kind: 用哪个摄像头
    :return: 不返回
    """
    cap = cv2.VideoCapture(camera_kind)  # VideoCapture()中参数是1，表示打开外接usb摄像头
    cv2.namedWindow('camera')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera', frame)
            key = cv2.waitKey(1)
            if key == ord('m'):
                # 手动拍照
                cv2.imwrite(save_path, frame)  # 保存路径
                break

            if key == ord('a'):
                # 自动拍照
                time.sleep(8)
                ret, frame = cap.read()
                cv2.imwrite(save_path, frame)  # 保存路径
                break

            if key == ord('k'):
                # 退出进程
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_name = 'test2.jpg'
    model_dir = str(r'D:\my_Academic\teacher_zyb\PaddleDetection\output\inference_model\ppyolo_mbv3_large_qr')
    image_file = os.path.join(str(r'D:\my_Academic\teacher_zyb\testfile'), img_name)
    camera(save_path=image_file, camera_kind=0)
