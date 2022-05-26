import numpy as np
import cv2
import os
from skimage import feature
import xgboost as xgb
import joblib



def gray_img_split(img, rows, cols):
    # 白黒画像を分割
    split_box=[]
    for row_img in np.array_split(img, rows, axis=0):
        for chunk in np.array_split(row_img, cols, axis=1):
            split_box.append(chunk)

    return split_box

def LBP_chair_top_mid(file_name):
    bgr_img = cv2.imread(file_name)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # LBPを用いたテクスチャ抽出（椅子）
    height, width = gray_img.shape
    for i in range(height):
        for j in range(width):
            if gray_img[i][j]>=250:
                gray_img[i][j]=255

    box = gray_img_split(gray_img, 3, 1)

    gray_img = np.concatenate([box[0], box[1]])

    points = 10
    radius = 15

    lbp = feature.local_binary_pattern(gray_img, points, radius)

    lbp_np = np.delete(np.float32(lbp).flatten(), np.where(np.float32(lbp).flatten() == 0))
    
    range_num=0
    for i in range(0,points):
        range_num+=(2**i)

    temp = cv2.calcHist([lbp_np],[0],None,[points],[0,range_num+1])
    temp = temp.ravel()
    print("chair_LBP_range{0}".format(range_num+1), "\n",temp)

    return temp


def create_lbp_model():
    save_path = os.getcwd() + "/static/models/"

    print('wood')

    wood = []
    for i in os.listdir(os.getcwd()+'/static/img/chair_wood'):
        file_name = os.getcwd()+'/static/img/chair_wood/{0}.jpg'.format(i.replace('.jpg',''))
        wood.append(LBP_chair_top_mid(file_name))

    print("leather")

    leather = []
    for i in os.listdir(os.getcwd()+'/static/img/chair_leather'):
        file_name = os.getcwd()+'/static/img/chair_leather/{0}.jpg'.format(i.replace('.jpg',''))
        leather.append(LBP_chair_top_mid(file_name))

    wood = np.array(wood)
    leather = np.array(leather)

    print(wood.shape)
    print(leather.shape)

    target_wood = np.array([0]*len(wood))
    target_leather = np.array([1]*len(leather))

    X = np.vstack((wood,leather))
    y = np.hstack((target_wood,target_leather))

    X = X.astype(float)
    y = y.astype(int)

    clf_bst_re = xgb.XGBClassifier(objective = "binary:logistic")
    clf_bst_re.fit(X, y)

    joblib.dump(clf_bst_re, save_path + "material_model.bin")


if __name__=="__main__":
    create_lbp_model()










