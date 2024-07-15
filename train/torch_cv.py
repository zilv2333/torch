import numpy as np
import torch
import torch.nn as nn
import pac.xyc_fuc as xyc
import cv2
import torch.nn.functional as F

net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),
                  nn.Sigmoid(),nn.AvgPool2d(kernel_size=2,stride=2),
                  nn.Conv2d(6,16,kernel_size=5),
                  nn.Sigmoid(),
                  nn.MaxPool2d(kernel_size=2,stride=2),
                  nn.Flatten(),
                  nn.Linear(16*5*5,120),
                  nn.Linear(120,84),
                  nn.Sigmoid(),
                  nn.Linear(84,10))

net.load_state_dict(torch.load("../Model/model.pt"))
net.eval()


tim=10
cap=cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret,frame=cap.read()

        if cv2.waitKey(tim) & 0xFF in[ ord('q'), ord(' ') ] :
            cv2.destroyAllWindows()
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gauss=cv2.GaussianBlur(gray,(5,5),0)

        _,thresh = cv2.threshold(gauss, 245,255,cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            epsilon = cv2.arcLength(cnt, True) * 0.5
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if 30000>cv2.contourArea(cnt) > 10000:
                # cv2.drawContours(thresh,approx,-1,(0,0,0),5)
                cv2.rectangle(thresh, (x + 5, y + 5), (x + w - 5, y + h - 5), (0, 0, 0), 2)
                im=thresh[x+50:x-50 + w, y+50:y-50 + h]
                im=cv2.resize(im,(28,28))
                im=torch.tensor(im,dtype=torch.float).reshape(-1,1,28,28)

                print(torch.argmax(net(im)))

            cv2.imshow('frame', thresh)





