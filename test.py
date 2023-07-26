import os
import os.path as osp
import json
import cv2

if __name__ == '__main__':
    print(osp.abspath(os.curdir))
    img = cv2.imread(r"F:\output.png")
    img = cv2.imread(r"E:\CBIBF3\storage\UnityProjects\UIHVizDemo\Assets\Samples\ileocecal_bbps1_000000_mini.png")
    print(img)
    cv2.imshow("png", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
