import cv2
import numpy as np
import math


cur_point = (0, 0)

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
# import pdb;pdb.set_trace()

img_output = np.zeros(img.shape, dtype=img.dtype)

start_point = (0, 0)
end_point = (100, 100)

r = 50


last_x, last_y = 0, 0
start_x, start_y = 0, 0
end_x, end_y = 0, 0


def MouseEvent(event, x, y, flags, param):
    global last_x, last_y
    global start_x, start_y
    global end_x, end_y
    if flags == cv2.EVENT_FLAG_LBUTTON:
        print('摁住左键 [{},{}]'.format(x, y))
        if last_x is 0 or last_y is 0:
            last_x, last_y = x, y

        img_output[(y - r):(y + r), (x - r): (x + r)] = img[(last_y - r):(last_y + r), (last_x - r): (last_x + r)]
        last_x, last_y = x, y

        cv2.imshow('22', img_output)
        # offset_x = int(cols / x)
        # offset_y = int(rows / y)
        # print(offset_x, offset_y)
        # if i+offset_y < rows and j+offset_x < cols:
        #     img_output[x,y] = img[(x+offset_y)%rows,(y+offset_x)%cols]
        # else:
        #     img_output[i,j] = 0


    # if event == cv2.EVENT_LBUTTONDOWN:
    #     print('down')
    #     start_x, start_y = x, y

    # if event == cv2.EVENT_LBUTTONUP:
    #     print('up')
    #     end_x, end_y = x, y
    #     print(start_x, start_y)
    #     print(end_x, end_y)
    #     s = np.array([start_x, start_y])
    #     e = np.array([end_x, end_y])
    #     d = np.sqrt(np.sum(np.square(e-s)))
    #     print(d)



for i in range(rows):
    for j in range(cols):
        if i < 100 and j < 200:
            img_output[i,j] = 0
        else:
            img_output[i,j] = img[i, j]
# import pdb;pdb.set_trace()

# cv2.imwrite("output.jpg", img_output)
# cv2.imshow('1111', img_output)
cv2.namedWindow('22')
cv2.setMouseCallback('22', MouseEvent)
cv2.imshow('22', img_output)
cv2.waitKey(0)
# while True:
#     cv2.imshow('22', img_output)
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 摁下q退出
#         break
# cv2.destroyAllWindows()
