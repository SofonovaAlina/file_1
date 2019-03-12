import os
import cv2
import shutil
import numpy as np
import time
import progressbar
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('path', metavar='path', type=str)

args = parser.parse_args()
path = os.path.join(os.getcwd(), args.path)

def image_preparing():
    cwd = os.getcwd()
    print (cwd)
    counter = 0
    images = ".png", ".jpg"
    videos = ".mp4", ".avi"
    path = 'tmp/'
    
    try:
        shutil.rmtree(path)
    except OSError:
        pass
    os.mkdir(path)

    for file in os.listdir(cwd):
        if file.endswith(videos):
            sum = 0
            #print (" file found", file)
            cap = cv2.VideoCapture(file)
            counter = counter+1

            ret, image_np = cap.read()
            with progressbar.ProgressBar(maxval=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as bar:
                while ret:
                    sum += 1
                    if sum > 20:
                        cv2.imwrite(os.path.join(path, (file[:-4]+"_%dframe.jpg" % sum)), image_np)
                    ret, image_np = cap.read()
                    bar.update(sum)

        if file.endswith(images):  
            shutil.copy(file, r"tmp")
            #print (" file found", file)
            counter = counter+1

    #print ("Total files found", counter)
    
    

def coord_search(image):
    
    ##-- Find the max-area contour
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ##-- Return contours coordinates 
    return cv2.boundingRect(cnt)

def alfa(image):
    src = image
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b ,g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst

def save_image(dst, file_name, path = None):
    if file_name.endswith("frame.png"):
        path = file_name[:file_name.rfind("_")]
        file_name = file_name[file_name.rfind("_")+1:]
        
    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + "\\" + file_name + ".png", dst)
    else:    
        cv2.imwrite(file_name + ".png", dst)

        
        
#-- Parameters
BLUR_KERNEL= 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0, 0.0, 0.0, 0.0)
hsv_min = np.array((40, 0, 0), np.uint8)
hsv_max = np.array((100, 255, 255), np.uint8)
kernel = np.ones((3,3),np.uint8)


def img_proc_color(name):
    img = cv2.imread("tmp/" + name, -1)
    img = cv2.GaussianBlur(img, (3,3), 1)

    #-- Normalization, mask creation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    thresh = cv2.erode(thresh, kernel, iterations = 1)
    thresh = cv2.bitwise_not(thresh)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    thresh = cv2.dilate(thresh, kernel, 1)
    thresh = cv2.bitwise_not(thresh)

    #-- Find contours and sort by area
    contour_info = []
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour
    #-- Mask is black, polygon is white
    mask = np.zeros(thresh.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    
    x,y,w,h = coord_search(thresh)
    
    #-- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations = MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations = MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR_KERNEL, BLUR_KERNEL), 0)
    mask_stack = np.dstack([mask]*4)

    img = alfa(img)
    
    #-- Blend masked img into MASK_COLOR background
    mask_stack  = mask_stack.astype('float32') / 255.0
    img         = img.astype('float32') / 255.0

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    masked = masked[y:y+h, x:x+w]
    return masked


def save_image(dst, file_name, path = "out/"):
    if file_name.endswith("frame.png"):
        path += os.path.join(file_name[:file_name.rfind("_")], "/")
        file_name = file_name[file_name.rfind("_")+1:]
        
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path + file_name[:-4] + ".png")
    cv2.imwrite(path, dst)
    
def main():
    print(path)
    os.chdir(path)                  #   Строка которая перемещает в директорию где лежат обрабатываемые файлы 
                                    #   в которой будет выполнятсья код
    image_preparing()               #   Подготовка файлов к нахождению объектов. Файлы пишутся во временную папку tmp
    for file in progressbar.progressbar(os.listdir("tmp")):          # Цикл с прогрессбаром для наглядности как выполняеся код
        image = img_proc_color(file)                                 # нахождение объекта на изображении
        save_image(image, file)                                      # Сохранение файла


    # Ниже мы удаляем временную папку tmp за ненадобностью    

    try:
        shutil.rmtree("tmp")
    except OSError:
        pass
    
main()