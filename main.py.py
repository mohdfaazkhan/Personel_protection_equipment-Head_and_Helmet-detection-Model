import os
import cv2
import time
import socket
#import pyodbc
import imageio
import imutils
import logging
import warnings
import torch
import threading
import subprocess
import numpy as np
import pandas as pd
#from sort import *
import smtplib, ssl
#import pymysql.cursors
# import tensorflow as tf
#from utils import label_map_util
from threading import Thread, Lock
import xml.etree.ElementTree as ET
from datetime import datetime,date

import traceback

#from utils import visualization_utils as vis_utils

from logging.handlers import TimedRotatingFileHandler

warnings.filterwarnings("ignore")

writer = cv2.VideoWriter('wobot_final_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'),30,(640,360))

class CameraStream(object):
    def __init__(self, src=0):
        #self.stream = cv2.VideoCapture("%s"%RSTP_protocal)
        self.stream = cv2.VideoCapture("Top 10 Safety Vest For Construction For Men And Women.mp4")

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
            time.sleep(.05)

    def read(self):
        try:
            self.read_lock.acquire()
            frame = self.frame.copy()
            self.read_lock.release()
            return frame
        except:
            pass

    def stop(self):
        self.started = False
        # logger.info("entered in stop function")         
        self.thread.join(timeout=1)
        # logger.info("exit from stop function")

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()   


def box_normal_to_pixel(box,dim):    
        width, height = dim[0], dim[1]
        box_pixel = [int(box[1]), int(box[0]), int(box[3]), int(box[2])]

        return np.array(box_pixel)

def unique_count_app(a):
    colors, count = np.unique(
            a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    print(colors)
    return colors[count.argmax()]

def get_mode(img):
    unq,count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
    return unq[count.argmax()]

def myfunction():   
               
    while video_capture:
    
        frame1 = video_capture.read()
        
        
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # print(frame1.shape)
        h,w = frame1.shape[0:2]    # 720*1280
        frame1 = cv2.resize(frame1,(640,int(h*(640/w))))
        # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
       
        
        #print(frame1.shape)
        #frame1 = cv2.resize(frame1,(640,640))
        image_np = frame1.copy()
        dim = image_np.shape[0:2]                                                                                                                
        #model Actual prediction
        pred_model = model(image_np)
        # print(pred_model.pandas().xyxy[0])
        df = pred_model.pandas().xyxy[0]
        boxes = pred_model.xyxy[0][:,:4].cpu()
        scores = pred_model.xyxy[0][:,4].cpu()
        classes = pred_model.xyxy[0][:,5].cpu() 
    
        # print("boxes: ",boxes)
        # print("scores: ",scores)
   
        # x1,x2,y1,y2 =  box_normal_to_pixel(boxes.tolist(),dim)
        # print("x1,x2,y1,y2: ",x1,x2,y1,y2)

        # for box in boxes.tolist():
        for i in range(df.shape[0]):
            # print(box[0],box[1],box[2],box[3])
        #     # x1,x2,y1,y2 =  box_normal_to_pixel(box,dim)
            # (y1, x1) = (box[0], box[1])
            # (y2, x2) = (box[2], box[3])
        #     print("x1,x2,y1,y2: ",x1,x2,y1,y2)
            cropimg=frame1[int(boxes[i][1]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][2])]
            # mycolor = np.average(cropimg, axis = (0,1))
            mycolor = get_mode(cropimg)
            # print(mycolor)
            # cv2.putText(frame1, str(df['name'][i]), (int(boxes[i][0]),int(boxes[i][1])),cv2.FONT_HERSHEY_COMPLEX,.5, (float(mycolor[0]),float(mycolor[1]),float(mycolor[2])),1)
            cv2.putText(frame1, str(df['name'][i]), (int(boxes[i][0]),int(boxes[i][1])),cv2.FONT_HERSHEY_COMPLEX,.5, (255,0,0),1)
            # cv2.putText(frame1, str(df['confidence'][i].round(decimals=2)*100), (int(boxes[i][0]+90),int(boxes[i][1])),cv2.FONT_HERSHEY_COMPLEX,.5, (float(mycolor[0]),float(mycolor[1]),float(mycolor[2])),1)
            cv2.putText(frame1, str(df['confidence'][i].round(decimals=2)*100), (int(boxes[i][0]+90),int(boxes[i][1])),cv2.FONT_HERSHEY_COMPLEX,.5, (255,0,0),1)
            # cv2.rectangle(frame1,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(255,0,0),2)
            cv2.putText(frame1, str('%'), (int(boxes[i][0]+125),int(boxes[i][1])),cv2.FONT_HERSHEY_COMPLEX,.5, (255,0,0),1)

            cv2.rectangle(frame1,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(float(mycolor[0]),float(mycolor[1]),float(mycolor[2])),2)
        
            

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',frame1)
        print(frame1.shape)

        
        writer.write(frame1)
        
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(9) & 0xFF == ord('q'):
            break
        

    # Closes all the frames
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    yolo_file_path = r'./yolov5'
    # model = torch.hub.load(yolo_file_path, 'custom', path= 'best.pt', source='local',force_reload = True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./runs/train/exp14/weights/best.pt', force_reload=True) 
    print("Model loaded Successfully")

    while True :
        video_capture = CameraStream().start()        
        while video_capture.stream.isOpened():
            try:            
                myfunction()
            except Exception as e :
                print("Main function is failing please check error is %s"%e)                
        e ="None frame encounterd or Video feed from camera is not available"

        video_capture.stop()
        print("video capture has been stoped")
        cv2.destroyAllWindows()
        print("window has been destroyed")
            
            
      