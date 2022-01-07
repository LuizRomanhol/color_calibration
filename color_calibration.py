import cv2
import numpy as np
import math

x_points = []
y_points = []
running = True
graytones = []
clicks = 0
radius = 0
houghsource = np.array([])

def hough(img):
    mask = np.clip(img-255,0,255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 10)
    #cv2.imshow('image',edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=100, maxLineGap=150)
    for line in lines:
	    x1, y1, x2, y2 = line[0]
	    cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return mask    
    
def click_event(event, x, y, flags, params):
 
    if event == cv2.EVENT_LBUTTONDOWN:
        global clicks
        global running
        clicks = clicks+1
        if clicks == len(graytones)+1:
            running = False
            return x,y

        t = 5
        color = (0,255,255)
        global houghsource
        houghimg = houghsource.copy()
        cv2.floodFill(houghimg, None, (x,y), color, loDiff = (t,t,t),upDiff = (t,t,t))
        rangeimg = cv2.inRange(houghimg,color,color)
        M = cv2.moments(rangeimg)

        X = round(M['m10'] / M['m00'])
        Y = round(M['m01'] / M['m00'])
        global x_points,y_points
        x_points.append(X)
        y_points.append(Y)
        global radius
        radius = int(math.sqrt(cv2.countNonZero(rangeimg))/2*0.75)
        
        cv2.circle(img,(X,Y),radius,(0,255,255), thickness=3)

        cv2.imshow('image',img)
        return x,y
 
def get_region_mean(img,point,radius):
    x, y = point
    sample = img[y-radius:y+radius, x-radius:x+radius]
    (rsquare, gsquare,bsquare) = cv2.split(samples)
    r, g, b = [int(rsquare.mean()),int(gsquare.mean()),int(bsquare.mean())]
    return (r,g,b)


def white_balance(img,x,y,radius):

    total = np.array([0,0,0])
    arr = [2,3]
    for i in arr:
        total = total + get_region_mean(img,(x[i],y[i]),radius)
    
    #grey = [((total/len(x)).mean())]*3
    grey = [((total/len(arr)).mean())]*3
    delta = (grey - total/len(arr)).astype(int)
    
    return np.clip(img.astype(int) + delta,0,255).astype('uint8')

def intensity_adjust(img,graytones,x,y,radius):
    gray = []
    print(x,graytones,y)

    for i in range(len(x)):
        gray.append(int(np.array(get_region_mean(img,(x[i],y[i]),radius)).mean()))
    
    z = np.polyfit(gray,graytones,3)
    p = np.poly1d(z)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pgray = np.clip(p(grayimg),0,255)
    delta = np.clip(pgray.astype(int) - grayimg.astype(int) + np.ones(grayimg.shape)*255,0,255*2)
    deltacolor = cv2.cvtColor(delta.astype('uint16'), cv2.COLOR_GRAY2RGB) 
    finalimg = np.clip(img.astype(int) + deltacolor.astype(int) - (np.ones(img.shape)*255).astype(int),0,255)

    return finalimg.astype('uint8')

def get_text_input():
    print("por favor digite o endereço da imagem\n")
    imgpath = input()
    #print("por favor digite os numeros dos tons de cinza da paleta de cores, separados por espaço")
    #graytones = input()
    
    if len(imgpath) == 0:
        imgpath = 'images/paleta.jpg'
    #if len(graytones) == 0:
    graytones = "243 200 161 120 85 52"

    graytones = graytones.split(" ")
    
    for i in range(len(graytones)):
        graytones[i] = int(graytones[i])

    return imgpath, graytones
 
def resize_image(img):
    maxsize = 800
    old_shape = img.shape
    for i in [0,1]:
        if img.shape[i]>maxsize:
            width = int(img.shape[1]*maxsize/img.shape[i])
            height = int(img.shape[0]*maxsize/img.shape[i])
            img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)
    new_shape = img.shape
    rfactor = (old_shape[0]/new_shape[0],old_shape[1]/new_shape[1])
    return img, rfactor

def get_image_points(img):

    backupimg = img.copy()
    global houghsource
    houghsource = hough(img.copy())
    while (running):
        cv2.setMouseCallback('image', click_event)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    x = x_points.copy()
    y = y_points.copy()
    global radius

    return x,y,radius, backupimg

def update_size(x,y,rfactor,imgpath,radius):
    
    x = (np.array(x)*rfactor[0]).astype(int)
    y = (np.array(y)*rfactor[1]).astype(int)
    radius = int(radius*(rfactor[0]+rfactor[1])/2)

    img = cv2.imread(imgpath, 1)
    return x,y,radius,img

if __name__=="__main__":

    imgpath, graytones = get_text_input()
    img = cv2.imread(imgpath, 1)
    img,rfactor = resize_image(img)

    cv2.imshow('image', img)

    x,y,radius,img = get_image_points(img)
    x,y,radius,img = update_size(x,y,rfactor,imgpath,radius)

    #print(x,y,graytones,"DEBUG")
    #img = intensity_adjust(white_balance(img,x,y,radius),graytones,x,y,radius)
#
    #img = intensity_adjust(img,graytones,x,y,radius)
    img = white_balance(img,x,y,radius)
    img = intensity_adjust(img,graytones,x,y,radius)

    cv2.imshow('image', resize_image(img)[0])

    cv2.waitKey(0)

    print("a imagem calibrada foi salva na pasta em que o programa foi executado")

    cv2.imwrite("result.jpg",img)
    cv2.destroyAllWindows()