import numpy as np
import random
import cv2
import imutils
# 讀取中文路徑圖檔，並轉換為BGR
def cv_imread(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

# 顯示圖檔
def show_img(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)

# Numpy儲存中文路徑圖檔
def cv_save(image, result_path):
    cv2.imencode('.jpg', image)[1].tofile(result_path)    
    
# 比例縮小圖檔
def resize_img(image, pos):
    image = cv2.resize(image, dsize=None, fx=pos, fy=pos)
    return image

# 旋轉圖片
def rotate_img(image):
    # 讀取圖片大小
    (h, w, d) = image.shape
    # 找到圖片中心
    center = (w // 2, h // 2)
    # 代表隨機順逆時針旋轉0-2度
    angle = random.randint(-40, 40) / 20
    # 縮放倍數為1.03倍，避免旋轉時小狗圖案被裁切掉
    M = cv2.getRotationMatrix2D(center, angle, 1.04)
    # (w, h )代表圖片縮放與旋轉後，需裁切成的尺寸
    image = cv2.warpAffine(image, M, (w, h))
    return image

# 漫水填充法去背
def image_matting(image):
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(image, mask, (200, 30), (255, 255, 255), 
                  (110, 110, 110), (30, 30, 30), 
                  cv2.FLOODFILL_FIXED_RANGE)#110 30   200, 30
    return image
def image_matting_hand(image):
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(image, mask, (30, 10), (255, 255, 255), 
                  (110, 110, 110), (30, 30, 30), 
                  cv2.FLOODFILL_FIXED_RANGE)#110 30
    return image
def min_rectangle(image2gray, image2):
    image2_copy2 =image2.copy()
    areaThreshold = 1000
    #binaryImage = cv2.threshold(image2gray, 254, 255, cv2.THRESH_BINARY)[1]
    show_img('image2_copy2', image2_copy2) 
    image2gray_inv = cv2.bitwise_not(image2gray)#反轉顏色 黑白互換
    #contours = cv2.findContours(binaryImage,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cv2.findContours(image2gray_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("cnts",cnts) 
    #cnts = contours[0]
    cnts = imutils.grab_contours(cnts)
    #print("cntsa",cnts) 
    print("aaa",len(cnts))
    for cnt in cnts:
        #print("cnt",cnt)
        if cv2.contourArea(cnt) < areaThreshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image2_copy2, (x,y), (x + w, y + h), (0, 255, 0), 2)#沒有方向角(正長方形)
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(image2_copy2, [box], 0, (0, 0, 255), 2)
    show_img("image2_copy2",image2_copy2)
    cv2.imwrite("./result/remove_back3.png",image2_copy2) 

# 前後景合成
def prduce_pic(image1, image2, x, y):
    image2_copy = image2.copy()
    
    # 前景上小狗圖像去背
    image2_copy = image_matting(image2_copy)
    
    show_img('a', image2_copy)
    cv2.imwrite("./result/remove_back.png",image2_copy)
    #image2_copy = image_matting_hand(image2_copy)#濾手
    # 前景上產生小狗圖像的mask
    image2gray = cv2.cvtColor(image2_copy, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./result/remove_back1.png",image2gray)
    show_img('ass', image2gray)    
    #ret, mask_thres = cv2.threshold(image2gray, 200, 255, cv2.THRESH_BINARY)#254 255
    mask_thres= cv2.adaptiveThreshold(image2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    show_img('ass', mask_thres) 
    min_rectangle(mask_thres, image2)
    show_img('ass', mask_thres)    
    cv2.imwrite("./result/remove_back2.png",mask_thres) 
    # 開運算去除mask中白色雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask_thres, cv2.MORPH_OPEN, kernel)
    #mask = cv2.dilate(mask_thres,mask)
    show_img('a',  mask)

    # 背景上定義小狗ROI區域
    rows, cols, channels = image2.shape
    roi = image1[y:y + rows, x:x + cols]
    #show_img('a_roi', roi)


    # 背景上ROI區域摳出小狗圖案mask
    image1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    show_img('a', image1_bg)

    # 前景上產生小狗圖像的inverse mask
    mask_inv = cv2.bitwise_not(mask)
    show_img('a', mask_inv)

    # 前景上小狗圖像的inverse mask內填充小狗圖案
    image2_fg = cv2.bitwise_and(image2, image2, mask=mask_inv)
    show_img('a', image2_fg)

    # 將「摳出小狗圖案mask的背景」與「填充小狗圖案的前景」相加
    dst = cv2.add(image1_bg, image2_fg)
    # show_img('dst', dst)

    # 用dst替換掉背景中含有小狗的區域
    image1[y:y + rows, x:x + cols] = dst
    # show_img('image1', image1)

    return image1

if __name__ == '__main__':
    #***須更改image_matting中cv2.floodFill的第二參數-->去背參考顏色座標***
    result_path = './result/'
    front_path = './0424test2/14.jpg'
    back_path = './background.jpg'

    front = cv_imread(front_path)
    front1 = cv2.resize(front,(277,369))#739 554 
    # show_img('front', front)

    for i in range(1):
        #front1 = rotate_img(front)
        #front1 = resize_img(front1, 0.9615)
        back = cv_imread(back_path)
        # show_img('back', back)
        result = prduce_pic(back, front1, random.randrange(0, 1), 
                            random.randrange(0, 1))
        cv_save(result, result_path + str(i) + '.jpg')
        print('  成功儲存第 {} 張圖片： {}'.format(i, str(i) + '.jpg'))
        print('=' * 50)
    print('※程式執行完畢')
