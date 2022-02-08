
import os
import cv2 
import Equirec2Perspec as E2P 
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
import numpy as np
import sys 
# Set recursion limit
sys.setrecursionlimit(10 ** 9)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class DraggablePolygon:
    lock = None
    def __init__(self, ax):
        print('__init__')
        self.press = None

        
        h = 10.
        w = 10.
        self.geometry = [[-0.5*h,-0.5*w], [0.5*h,-0.5*w], [0.5*h, 0.5*w], [-0.5*h, 0.5*w]]
        self.newGeometry = []
        poly = plt.Polygon(self.geometry, closed=True, fill=False, linewidth=3, color='#F97306')
        ax.add_patch(poly)
        self.poly = poly

    def connect(self):
        'connect to all the events we need'
        print('connect')
        self.cidpress = self.poly.figure.canvas.mpl_connect(
        'button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect(
        'button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect(
        'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        print('on_press')
        if event.inaxes != self.poly.axes: return
        if DraggablePolygon.lock is not None: return
        contains, attrd = self.poly.contains(event)
        if not contains: return

        if not self.newGeometry:
            x0, y0 = self.geometry[0]
        else:
            x0, y0 = self.newGeometry[0]

        self.press = x0, y0, event.xdata, event.ydata
        DraggablePolygon.lock = self

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggablePolygon.lock is not self:
            return
        if event.inaxes != self.poly.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        xdx = [i+dx for i,_ in self.geometry]
        ydy = [i+dy for _,i in self.geometry]
        self.newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly.set_xy(self.newGeometry)
        self.poly.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        print('on_release')
        if DraggablePolygon.lock is not self:
            return

        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry
        print(np.mean(np.array(self.geometry), axis=0))


    def disconnect(self):
        'disconnect all the stored connection ids'
        print('disconnect')
        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)


#dp = DraggablePolygon(ax=ax2)
#dp.connect()
#print(dp.geometry)



if __name__ == '__main__':
    
   
    
    
    class Rect : 
        def __init__(self, pt1, pt2):
            self.pt1 = pt1
            self.pt2 = pt2
            
    
    def DragRect(event, x, y, flags, params):
        global ix, iy, drag, myRect, imConcat, copy
        
        if (event == cv2.EVENT_LBUTTONDOWN):
            if ((x >= myRect.pt1[0])  and( x<= myRect.pt2[0]) and (y >= myRect.pt1[1]) and( y<= myRect.pt2[1])) : 
                drag = True
                ix,iy = x,y
            else : drag = False
            #print(drag)
            
        if (event == cv2.EVENT_MOUSEMOVE):
            if (drag) :
                pass
                
                
                
  
                offset  = int(0.5*(wc - w2))
                
                """
                xmin = offset + 0.25*wr
                xmax = offset + w2 - 0.25*wr
                
                ymin = h1 + 0.25*hr
                ymax = h1 + h2 - 0.25*hr
                
                if (x > xmax) : x = xmax
                if (x < xmin) : x = xmin
                if (y > ymax) : y = ymax
                if (y < ymin) : y = ymin
                """
                dx = x-ix
                dy = y-iy
                
                pt1 = (myRect.pt1[0] + dx, myRect.pt1[1] + dy)
                pt2 = (myRect.pt2[0] + dx, myRect.pt2[1]+ dy)

                lat = int((0.5*(pt1[0]+pt2[0]) - (offset + w2*0.5)) / (w2*0.5) *180)
                lon = int(-(0.5*(pt1[1]+pt2[1]) - (h1 + h2*0.5)) / (h2*0.5) *90)
                
                img = equ.GetPerspective(FOV, lat , lon, h1, w1) # Specify parameters(FOV, theta, phi, height, width)
                
 
                
                offset  = int(0.5*(wc - w1))
                imConcat[:h1,offset:w1 + offset, : ] = img

                copy = imConcat.copy()
                
                cv2.rectangle(copy, pt1=pt1, pt2=pt2, color=rectColor, thickness=2)
                
                
        if (event == cv2.EVENT_LBUTTONUP):
            if (drag) :
  
                dx = x-ix
                dy = y-iy
                
                pt1 = (myRect.pt1[0] + dx, myRect.pt1[1] + dy)
                pt2 = (myRect.pt2[0] + dx, myRect.pt2[1]+ dy)
                print(pt1)
                print(pt2)
                
                myRect = Rect(pt1=(myRect.pt1[0] + dx, myRect.pt1[1] + dy), pt2=(myRect.pt2[0] + dx, myRect.pt2[1]+ dy))
                
            drag = False


    equiImg = cv2.imread('src/image.jpg', cv2.IMREAD_COLOR)
    equ = E2P.Equirectangular(equiImg)    # Load equirectangular image
    (h2, w2, n)  = equiImg.shape
    
    
    

    FOV = 20
    resolution = 400
    ratio =  16./9.0
    
    
    
    h1 = resolution
    w1 = int(resolution * ratio) 
    
    resSrc = int(w1)
    h2 = int(resSrc*h2/w2)
    w2 = resSrc
    
    

    
    
    
    
    srcImg = cv2.resize(equiImg, (w2,h2), interpolation=cv2.INTER_CUBIC)
    
   
    #
    # FOV unit is degree 
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension 
    #
    
    rectColor = (0,0,0)
    
    hc = h1+h2
    wc = np.max([w1,w2])
    
    
    hr = int(resSrc/10 * FOV / 60. )
    wr = int(hr * 16.0 / 9.0)
    
    offset  = int(0.5*(wc - w2))
    
    ix = int(offset + w2*0.5-0.5*wr)
    iy = int(h1 + 0.5*h2 - 0.5*hr)
    
    myRect = Rect(pt1 = (ix,iy), pt2 = (ix+ wr, iy+hr))
    
                   
    pt1 = (myRect.pt1[0] , myRect.pt1[1])
    pt2 = (myRect.pt2[0] , myRect.pt2[1])
    lat = int((0.5*(pt1[0]+pt2[0]) - (offset + w2*0.5)) / (w2*0.5) *180)
    lon = int(-(0.5*(pt1[1]+pt2[1]) - (h1 + h2*0.5)) / (h2*0.5) *90)

    img = equ.GetPerspective(FOV, lat, lon, h1, w1) # Specify parameters(FOV, theta, phi, height, width)
    
    
    imConcat = np.zeros((hc, wc, 3),dtype=np.uint8)
    imConcat[:,:,0] = np.ones([hc, wc])*255
    imConcat[:,:,1] = np.ones([hc, wc])*255
    imConcat[:,:,2] = np.ones([hc, wc])*255
    
    offset  = int(0.5*(wc - w1))
    imConcat[:h1,offset:w1 + offset, : ] = img
    
    offset  = int(0.5*(wc - w2))

    imConcat[h1:,offset:w2 + offset, : ] = srcImg
 
    windowName = 'named window'
    cv2.imshow( windowName, imConcat )
   
    copy = imConcat.copy()
    drag = False
    
    
    copy = imConcat.copy()
                
    cv2.rectangle(copy, pt1=pt1, pt2=pt2, color=rectColor, thickness=2)
                
    
    
    while(True):

        cv2.imshow(windowName, copy)
        cv2.setMouseCallback(windowName, DragRect)
        key = cv2.waitKey(30) & 0xFF
        # press 'r' to reset the window
        if key == ord("r"):
            copy = imConcat.copy()
        if (key == ord('+')) : 
            FOV -= 10
            if (FOV < 10) : FOV = 10
            hr = int(resSrc/10 * FOV / 60. )
            wr = int(hr * 16.0 / 9.0)
            
            pt1 = myRect.pt1
            pt2 = myRect.pt2
            
            pt1 = (int(0.5*(pt1[0]+pt2[0]) -0.5*wr), int(0.5*(pt1[1]+pt2[1]) - 0.5*hr ))
            pt2 = (int(0.5*(pt1[0]+pt2[0]) +0.5*wr), int(0.5*(pt1[1]+pt2[1]) + 0.5*hr ))
            
            myRect = Rect(pt1=pt1, pt2=pt2)
            
            lat = int((0.5*(pt1[0]+pt2[0]) - (offset + w2*0.5)) / (w2*0.5) *180)
            lon = int(-(0.5*(pt1[1]+pt2[1]) - (h1 + h2*0.5)) / (h2*0.5) *90)
                
            img = equ.GetPerspective(FOV, lat , lon, h1, w1) # Specify parameters(FOV, theta, phi, height, width)
             
            offset  = int(0.5*(wc - w1))
            imConcat[:h1,offset:w1 + offset, : ] = img
            
            
            copy = imConcat.copy()
            cv2.rectangle(copy, pt1=pt1, pt2=pt2, color=rectColor, thickness=2)
            
            print(FOV)
        if (key == ord('-')) : 
            FOV += 10
            if (FOV > 150) : FOV = 150
            hr = int(resSrc/10 * FOV / 60. )
            wr = int(hr * 16.0 / 9.0)
            
            pt1 = myRect.pt1
            pt2 = myRect.pt2
            
            pt1 = (int(0.5*(pt1[0]+pt2[0]) -0.5*wr), int(0.5*(pt1[1]+pt2[1]) - 0.5*hr ))
            pt2 = (int(0.5*(pt1[0]+pt2[0]) +0.5*wr), int(0.5*(pt1[1]+pt2[1]) + 0.5*hr ))
            
            myRect = Rect(pt1=pt1, pt2=pt2)
            lat = int((0.5*(pt1[0]+pt2[0]) - (offset + w2*0.5)) / (w2*0.5) *180)
            lon = int(-(0.5*(pt1[1]+pt2[1]) - (h1 + h2*0.5)) / (h2*0.5) *90)
            img = equ.GetPerspective(FOV, lat , lon, h1, w1) # Specify parameters(FOV, theta, phi, height, width)
             
            offset  = int(0.5*(wc - w1))
            imConcat[:h1,offset:w1 + offset, : ] = img
            copy = imConcat.copy()
            cv2.rectangle(copy, pt1=pt1, pt2=pt2, color=rectColor, thickness=2)
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # close all open windows
    cv2.destroyAllWindows() 