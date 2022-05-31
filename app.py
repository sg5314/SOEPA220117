from flask import Flask, request, render_template, redirect
import time
import os
import numpy as np
import cv2
import itertools
import joblib
from skimage import feature
from datetime import datetime
import string
import random
# https://note.nkmk.me/python-collections-counter/
from collections import Counter



# ページに表示する「机」の画像のパス
dir_list = sorted(os.listdir(os.getcwd() +'/static/img/table_imgs/'))

data_result_path = os.getcwd() + '/static/datasets_results/'

database_result_path = os.getcwd() + '/static/'

# 家具の画像が格納されてるパス
database_imgs = os.getcwd() + '/static/img/'

def data_list(temp):
    return sorted(os.listdir(temp))

def make_data(target_data, target_name, target_label = 'table'):

    # 選択した机とデータベースに存在する家具（椅子，キャビネット）の直積集合を作成

    # 調査対象データ
    target_data = np.hstack((target_data, target_name))

    # 椅子のデータベース
    chair_datas = np.loadtxt(database_result_path + 'chair_database_results/chair_dataset.csv',delimiter=",")
    chair_datas = chair_datas * 100
    temp = data_list(database_imgs+'chair_imgs/')
    chair_names = np.array(temp).reshape(len(temp),-1)

    chair_datas = np.hstack((chair_datas, chair_names))


    # 机のデータベース
    table_datas = np.loadtxt(database_result_path + 'table_database_results/table_dataset.csv',delimiter=",")
    table_datas = table_datas * 100
    temp = data_list(database_imgs+'table_imgs/')
    table_names = np.array(temp).reshape(len(temp),-1)

    table_datas = np.hstack((table_datas, table_names))


    # キャビネットのデータベース
    cabinet_datas = np.loadtxt(database_result_path + 'cabinet_database_results/cabinet_dataset.csv',delimiter=",")
    cabinet_datas = cabinet_datas * 100
    temp = data_list(database_imgs+'cabinet_imgs/')
    cabinet_names = np.array(temp).reshape(len(temp),-1)

    cabinet_datas = np.hstack((cabinet_datas, cabinet_names))

    if target_label == 'chair':
        cartesian_product = list(itertools.product([target_data], table_datas, cabinet_datas))
    elif target_label == 'table':
        cartesian_product = list(itertools.product(chair_datas, [target_data], cabinet_datas))
    elif target_label == 'cabinet':
        cartesian_product = list(itertools.product(chair_datas, table_datas, [target_data]))

    # イテレータになっているので配列に入れなおす（cartesian_product）
    for con, car_pro in enumerate(cartesian_product):
        car_list = np.array([])
        for car in car_pro:
            car_list = np.append(car_list, car)
        cartesian_product[con] = car_list
    

    for con2, pro in enumerate(cartesian_product):
        num_list = []
        name_list = []
        for con3, div_data in enumerate(pro):
            if con3 != 9 and con3 != 18 and con3 != 27:
                num_list.append(div_data)
            else:
                name_list.append(div_data)
        num_list = np.append(num_list,name_list)
        cartesian_product[con2] = num_list


    cartesian_product = np.array(cartesian_product)

    return cartesian_product

def cos_sim(v1, v2):
    """
    cos類似度
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def same_rate_deter(scene_name, feature_num, No1_features, No2_features, No3_features):
    """
    各1位，2位，3位において同率が入っている場合の対処法

    学習データから，10個ランダムに，抽出．（データ：椅子，机，キャビネットによる特徴量の組み合わせ）
    抽出した10個のランダム特徴量データ集合を「検査データ集合」とする．
    同率の集合の要素1つ１つにおいて，検査データ集合の各要素とcos類似度を算出
    ＞10個のcos類似度の総和を算出

    結果，最もcos類似度の総和が大きかった，同率の集合の要素を
    各順位の優勝組み合わせとする

    scene_name：シーン名，feature_num：特徴量次元数
    No1_features, No2_features, No3_features：各順位の特徴量リスト
    """

    print("-------same_rate_deter関数スタート-------", flush=True)

    data_path = os.getcwd() + '/static/learn_data/'

    # 検査データ集合の作成
    # データをロード
    if scene_name == "modern_living":
        modern_living = np.loadtxt(data_path + 'living_modern_result/img_dataset.csv', delimiter=",")# (37, 25)

        test_data_set = modern_living[np.random.choice(np.arange(0,len(modern_living)), 10, replace=False)]# 重複なし(10, 25)

    elif scene_name == "modern_office":
        modern_office = np.loadtxt(data_path + 'office_modern_result/img_dataset.csv', delimiter=",")# (32, 25)

        test_data_set = modern_office[np.random.choice(np.arange(0,len(modern_office)), 10, replace=False)]# 重複なし(10, 25)

    elif scene_name == "cute_living":
        cute_room = np.loadtxt(data_path + 'cute_room_result/img_dataset.csv', delimiter=",")# (26, 25)

        test_data_set = cute_room[np.random.choice(np.arange(0,len(cute_room)), 10, replace=False)]# 重複なし(10, 25)

    print("検査データ作成終了", flush=True)

    print("----検査スタート----",flush=True)
    # 順位の同率の集合の要素数1は，そのまま返却
    if len(No1_features) == 1:
        # 順位の同率の集合の要素数1は，そのまま返却
        re_no1_feature = No1_features[0]
    else:
        # 順位の同率の集合の要素数が複数
        no1_cos = [0]*len(No1_features)
        for che_num, no1_fea in enumerate(No1_features[:,:-3].astype(np.float32)):
            temp = 0
            for test in test_data_set:
                temp+=cos_sim(no1_fea[0:6], test[0:6])# chair HOG
                temp+=cos_sim(no1_fea[7:9], test[7:9])# chair CIELUV
                #temp+=cos_sim(no1_fea[9:15], test[9:15])# table HOG
                #temp+=cos_sim(no1_fea[15:17], test[15:17])# table CIELUV
                temp+=cos_sim(no1_fea[17:23], test[17:23])# cabinet HOG
                temp+=cos_sim(no1_fea[23:25], test[23:25])# cabinet CIELUV
            no1_cos[che_num]=temp
        no1_cos=np.array(no1_cos)
        re_no1_feature=No1_features[np.argmax(no1_cos)]
    print("--1位の検査終了--", flush=True)

    if len(No2_features) == 1:
        # 順位の同率の集合の要素数1は，そのまま返却
        re_no2_feature = No2_features[0]
    else:
        # 順位の同率の集合の要素数が複数
        no2_cos = [0]*len(No2_features)
        for che_num, no2_fea in enumerate(No2_features[:,:-3].astype(np.float32)):
            temp = 0
            for test in test_data_set:
                temp+=cos_sim(no2_fea[0:6], test[0:6])# chair HOG
                temp+=cos_sim(no2_fea[7:9], test[7:9])# chair CIELUV
                #temp+=cos_sim(no2_fea[9:15], test[9:15])# table HOG
                #temp+=cos_sim(no2_fea[15:17], test[15:17])# table CIELUV
                temp+=cos_sim(no2_fea[17:23], test[17:23])# cabinet HOG
                temp+=cos_sim(no2_fea[23:25], test[23:25])# cabinet CIELUV
            no2_cos[che_num]=temp
        no2_cos=np.array(no2_cos)
        re_no2_feature=No2_features[np.argmax(no2_cos)]
    print("--2位の検査終了--", flush=True)

    if len(No3_features) == 1:
        # 順位の同率の集合の要素数1は，そのまま返却
        re_no3_feature = No3_features[0]
    else:
        # 順位の同率の集合の要素数が複数
        no3_cos = [0]*len(No3_features)
        for che_num, no3_fea in enumerate(No3_features[:,:-3].astype(np.float32)):
            temp = 0
            for test in test_data_set:
                temp+=cos_sim(no3_fea[0:6], test[0:6])# chair HOG
                temp+=cos_sim(no3_fea[7:9], test[7:9])# chair CIELUV
                #temp+=cos_sim(no3_fea[9:15], test[9:15])# table HOG
                #temp+=cos_sim(no3_fea[15:17], test[15:17])# table CIELUV
                temp+=cos_sim(no3_fea[17:23], test[17:23])# cabinet HOG
                temp+=cos_sim(no3_fea[23:25], test[23:25])# cabinet CIELUV
            no3_cos[che_num]=temp
        no3_cos=np.array(no3_cos)
        re_no3_feature=No3_features[np.argmax(no3_cos)]
    print("--3位の検査終了--", flush=True)

    return re_no1_feature, re_no2_feature, re_no3_feature

def extract_match_evaluation(check_datas, scene_name):

    """
    （マッチする組み合わせを抽出）
    3．直積集合をモデルに投入     
    4．ランキング化

    check_datas：ユーザが選択した（机）とデータベースに存在する（椅子）と（キャビネット）の直積集合データ
    scene_name：ユーザが選択したシーン名
    """
    
    # 組み合わせデータを，（名前）と（特徴量）に分割
    check_datas_num = check_datas[:,0:25]
    check_datas_num=check_datas_num.astype(np.float32)
    check_datas_name = check_datas[:,25:28]


    if scene_name == "modern_living":
        # 学習済みモデルのダウンロード
        clf_bst_re = joblib.load(database_result_path + "models/modern_not_modern_living.bin")
        
    elif scene_name == "modern_office":
        # 学習済みモデルのダウンロード
        clf_bst_re = joblib.load(database_result_path + "models/modern_not_modern_office.bin")
        
    elif scene_name == "cute_living":
        # 学習済みモデルのダウンロード
        clf_bst_re = joblib.load(database_result_path + "models/cute_not_cute_room.bin")
    else:
        print('\033[31m' + "そのようなシーンは存在しません" + '\033[0m')
        clf_bst_re = None
        return [], [], []

    # クラス1（嫌い）に属する確率
    test_predict = clf_bst_re.predict_proba(check_datas_num)
    
    data_proba = np.c_[test_predict, check_datas]

    proba_sort = data_proba[np.argsort(data_proba[:,0].astype(np.float32))[::-1],:]

    top3_count_keys = list(Counter(proba_sort[:,0].astype(np.float32)))[:3]
    
    # 各1位，2位，3位において，同率の順位の数
    No1_count_num = Counter(proba_sort[:,0].astype(np.float32))[top3_count_keys[0]]
    No2_count_num = Counter(proba_sort[:,0].astype(np.float32))[top3_count_keys[1]]
    No3_count_num = Counter(proba_sort[:,0].astype(np.float32))[top3_count_keys[2]]

    print("同率No1の数：",No1_count_num)
    print("同率No2の数：",No2_count_num)
    print("同率No3の数：",No3_count_num)

    No1_sameRate_features = proba_sort[0:No1_count_num,2:] # 好きと嫌いの確率を除去
    No2_sameRate_features = proba_sort[No1_count_num:No1_count_num+No2_count_num,2:]# 好きと嫌いの確率を除去
    No3_sameRate_features = proba_sort[No1_count_num+No2_count_num:No1_count_num+No2_count_num+No3_count_num,2:]# 好きと嫌いの確率を除去

    no1_feature_imgName, no2_feature_imgName, no3_feature_imgName= same_rate_deter(scene_name, 25, 
                No1_sameRate_features, No2_sameRate_features, No3_sameRate_features)
    
    
    no1_feature_imgName = proba_sort[0]
    no2_feature_imgName = proba_sort[1]
    no3_feature_imgName = proba_sort[2]

    print('first>>>>>>>>>>>>>\n',no1_feature_imgName)
    print('second>>>>>>>>>>>>>\n',no2_feature_imgName)
    print("third>>>>>>>>>>>>>>\n",no3_feature_imgName)

    return no1_feature_imgName, no2_feature_imgName, no3_feature_imgName

def recomend_items(select_img_name, scene_name):

    """
    推薦プログラムを実行
    scene_name：ユーザが選択したシーン
    select_img_name：ユーザが選択した机番号

    ～流れ～
    1．直積集合を作成
    2．Pickleデータで，モデルをダウンロード
    3．直積集合をモデルに投入
    4．ランキング化
    """

    # 時間計測（開始）
    start = time.time()

    # 1．直積集合を作成
    # 取り出したい机のインデックス番号
    select_img_index = int(select_img_name.replace('.jpg', ''))
    print("select_img_index：", select_img_index)
    #target_table_index = 1

    table_datas = np.loadtxt(database_result_path + 'table_database_results/table_dataset.csv',delimiter=',')# ファイルのパスを指定
    table_datas = table_datas * 100
    table_datas = table_datas.tolist()
    # 選択した家具（机）の特徴量データを抽出
    target_table_data = table_datas[select_img_index]

    # 直積集合データ
    check_datas = make_data(target_table_data, select_img_name , target_label = 'table')
    print('直積集合データ作成終了', flush=True)
    
    # 3．直積集合をモデルに投入     4．ランキング化
    # 最もマッチする上位3つの組み合わせ　を出力
    top_1_matches, top_2_matches, top_3_matches  = extract_match_evaluation(check_datas, scene_name)

    if len(top_1_matches) == len(top_2_matches) == len(top_3_matches) == 0:
        return [], [], []

    # 時間計測（終了）
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]", flush=True)
    # 結果出力（椅子&机&キャビネット）
    print(top_1_matches[-3:], top_2_matches[-3:], top_3_matches[-3:])

    return top_1_matches[-3:], top_2_matches[-3:], top_3_matches[-3:]

def gray_img_split(img, rows, cols):
    # 白黒画像を分割
    split_box=[]
    for row_img in np.array_split(img, rows, axis=0):
        for chunk in np.array_split(row_img, cols, axis=1):
            split_box.append(chunk)

    return split_box

def LBP_chair_top_mid(gray_img):
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

def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])


app = Flask(__name__)

@app.route("/recommend", methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':

        select_img_name = request.form.get('submit-form')
        print('選択した机画像', select_img_name)
        select_scene_name = request.form.get('cpipr08')
        print('選択したシーン',select_scene_name)

        
        # 推薦プログラム実行　　　（top_1_names, top_2_names, top_3_names　＞＞　上位1位～上位3位までの家具の組み合わせ）
        top_1_names, top_2_names, top_3_names = recomend_items(select_img_name, select_scene_name)

        if len(top_1_names) == len(top_2_names) == len(top_3_names) == 0:
            return "Error"
        

        # 画面遷移　3秒間待機
        # time.sleep(3)

        print(top_1_names, top_2_names, top_3_names)

        recom_dic = {}
        recom_dic["no1_chair"] = top_1_names[0]
        recom_dic["no1_cabinet"] = top_1_names[2]
        recom_dic["no2_chair"] = top_2_names[0]
        recom_dic["no2_cabinet"] = top_2_names[2]
        recom_dic["no3_chair"] = top_3_names[0]
        recom_dic["no3_cabinet"] = top_3_names[2]

        if select_scene_name == "modern_living":
            select_scene_strname = "モダン風"
        elif select_scene_name == "modern_office":
            select_scene_strname = "モダン風"
        elif select_scene_name == "cute_living":
            select_scene_strname = "キュート"        

        return render_template('recommend.html', select_scene_strname = select_scene_strname, recom_dic = recom_dic, select_img_name = select_img_name)  #test



    else:
        return render_template('select_items_page.html', dir_list = dir_list)


@app.route('/')
def index():
    return render_template('select_items_page.html', dir_list = dir_list)



# 材質特徴量のページ
@app.route('/material', methods=['POST', 'GET'])
def materiality():
    if request.method == 'POST':
        stream = request.files['photo'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        #cv2.imshow('color', img) #この時点ではウィンドウは表示されない
        #cv2.waitKey(0) #ここで初めてウィンドウが表示される

        #time.sleep(5)
        ori_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cha_lbp = LBP_chair_top_mid(ori_img_gray)

        print("調査対象-椅子画像LBP\n",cha_lbp)

        clf_bst_re = joblib.load(database_result_path + "models/material_model.bin")

        print("予測結果")
        print("木目調 | 革調")
        y_pred = clf_bst_re.predict([cha_lbp])
        
        if y_pred[0] == 0:
            ans = "木目調のイス"
        else:
            ans = "革調のイス"

        # 画像保存
        dt_now = datetime.now().strftime("%Y_%m") + random_str(7)
        save_path = os.path.join(os.getcwd()+"/static/img/00_result_imgs/", dt_now + ".jpg")
        import pathlib
        pathlib.Path(save_path)
        cv2.imwrite(save_path, img)

        print("save", save_path)



        return render_template('select_items2_page.html', pre_chair = ans, img_name = dt_now + ".jpg")

    else:
        global now_scene_int
        now_scene_int=0
        return render_template('select_items2_page.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)            