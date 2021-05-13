from Detector_pytorch import Detector
import cv2
import time

def process_output(idx_, history):
    max_hist_len = 10  # 预测结果缓冲区

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history  # 返回本帧结果和历史结果

def main():

    det = Detector()
    cap = cv2.VideoCapture("test.mp4")
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    while True:
        _, im = cap.read()
        if im is None:
            break
        time1 = time.time()

        result = det.feedCap(im)
        result_im = result['frame']                  # deepsort人像部分

        time2 = time.time()
        time3 = time2-time1

        print('Done. (%.4fs)' % time3)
        cv2.imshow("demo", result_im)
        cv2.waitKey(t)

        if cv2.getWindowProperty("demo", cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()