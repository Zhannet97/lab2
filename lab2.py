import cv2, time
import numpy as np

def encode(capture, separator):
    f = open("EncodedVideo", "wb")

    start_time = time.time()

    ret, frame1 = capture.read()

    f.write(frame1.tobytes())
    f.write(separator.encode())
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    k = 0;
    while (1):
        k += 1
        ret, frame2 = capture.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f.write(flow.tobytes())
        f.write(separator.encode())
        if k == 100 or not ret:
            break
        prvs = next

    capture.release()
    f.close()
    print("Video successfully encoded in {} seconds\n".format(time.time()-start_time))

def decode(file,separator):
    f = open(file,"rb")
    encodedInfo = f.read()
    f.close()
    list = encodedInfo.split(separator.encode())

    frame1 = np.frombuffer(list[0],dtype=np.dtype('uint8')).reshape((848, 464, 3))

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    for item in list[1:-1]:
        flow = np.frombuffer(item,dtype=np.float32).reshape((848, 464, 2))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('Decoded Video', rgb)
        cv2.waitKey(10) & 0xff

    cv2.destroyAllWindows()


cap = cv2.VideoCapture("zhuzhu.avi")
separator = "NewItem"
encode(cap,separator)
decode("EncodedVideo",separator)