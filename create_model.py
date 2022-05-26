import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pydotplus
from sklearn.externals.six import StringIO
from scipy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import json
import graphviz
import joblib
import os
import re
from sklearn.model_selection import cross_val_score, KFold
from collections import Counter


def using_xgboost_Grid_Search(X_train, Y_train, save_path, file_name):

    #初期値設定
    clf_bst = xgb.XGBClassifier(objective = "binary:logistic")

    # verbose : ログ出力レベル
    # パラメーターを設定
    param_grid = {'max_depth':list(range(5,10)),
                'n_estimators':[50,60,70,80,90,100],
                }

    grid_search_xg = GridSearchCV(estimator = clf_bst, 
                                param_grid = param_grid,
                                scoring = 'accuracy',verbose=1, return_train_score=True,cv=3)# 
    
    grid_search_xg.fit(X_train, Y_train)

    print('ベストパラメータ：', grid_search_xg.best_params_)
    #print('ベストスコア（検証用データの平均正解率）：',grid_search_xg.best_score_)

    #学習
    clf_bst_re = xgb.XGBClassifier(max_depth = grid_search_xg.best_params_['max_depth'], 
                                    n_estimators = grid_search_xg.best_params_['n_estimators'], 
                                    objective = "binary:logistic") # **grid_search_xg.best_params_
    clf_bst_re.fit(X_train, Y_train)

    """# 予測
    test_predict = clf_bst_re.predict(X_test)

    print(np.sum(test_predict==Y_test),'/',len(Y_test))
    print('XGBoost + GridSearch  Accuracy:', accuracy_score(Y_test, test_predict))"""
    
    # 保存
    joblib.dump(clf_bst_re, save_path + file_name +".pickle")

def load_data(data_name, label_num):
    """
    Load csv file
    """
    learn_data = np.loadtxt(data_name, delimiter=",")
    label = np.full(len(learn_data),label_num) #第2引数で第1引数を初期化
    return learn_data, label


def create_model(modern_office_flag=True, modern_living_flag=True ,cute_room_flag=True):
    
    #データのパス
    data_path = os.getcwd() + '/static/learn_data/'
    save_path = os.getcwd() + "/static/models/"

    print('モデルの構築を行います　保存先：', save_path)

    if modern_office_flag:
        # モダンなオフィスルーム モダンでないオフィスルーム
        print('--------------------------------------------------------------------------', flush=True)
        # データを用意
        modern_office, modern_office_label = load_data(data_path + 'office_modern_result/img_dataset.csv',0)
        modern_office = modern_office * 100
        not_modern_office, not_modern_office_label = load_data(data_path + 'not_office_modern_result/img_dataset.csv', 1)
        not_modern_office = not_modern_office * 100

        #X (行＝scene番号，列＝特徴量)　行方向に連結
        X_EX1 = np.vstack((modern_office, not_modern_office))
        #Y　列方向に連結　1列
        Y_EX1 = np.hstack((modern_office_label, not_modern_office_label))

        # int型に変換    （nan = -2147483648）
        X_EX1 = X_EX1.astype(float)
        Y_EX1 = Y_EX1.astype(float) #int型をやめてfloat型に変更

        """# 学習データとテストデータに分割
        X_EX1_train, X_EX1_test, Y_EX1_train, Y_EX1_test = train_test_split(X_EX1, Y_EX1, test_size=0.3, random_state=0)#random_state = ?

        Train_counter = Counter(Y_EX1_train)
        Test_counter = Counter(Y_EX1_test)

        print('\n','モダンなリビングルーム　の学習用データ数：',Train_counter[0],'    ','モダンでないリビングルーム　の学習用データ数：',Train_counter[1], flush=True)
        print('モダンなリビングルーム　のテスト用データ数：',Test_counter[0],'    ','モダンでないリビングルーム　のテスト用データ数：',Test_counter[1],'\n', flush=True)"""

        using_xgboost_Grid_Search(X_EX1, Y_EX1, save_path, "modern_not_modern_office")

        print('--------------------------------------------------------------------------', flush=True)
    
    if modern_living_flag:
        # モダンなリビングルーム　モダンでないリビングルーム
        print('--------------------------------------------------------------------------', flush=True)
        # データを用意
        modern_living, modern_living_label = load_data(data_path + 'living_modern_result/img_dataset.csv',0)
        modern_living = modern_living * 100
        not_modern_living, not_modern_living_label = load_data(data_path + 'not_living_modern_result/img_dataset.csv', 1)
        not_modern_living = not_modern_living * 100

        #X (行＝scene番号，列＝特徴量)　行方向に連結
        X_EX2 = np.vstack((modern_living, not_modern_living))
        #Y　列方向に連結　1列
        Y_EX2 = np.hstack((modern_living_label, not_modern_living_label)) 

        # int型に変換    （nan = -2147483648）
        X_EX2 = X_EX2.astype(float)
        Y_EX2 = Y_EX2.astype(float) #int型をやめてfloat型に変更

        """# 学習データとテストデータに分割
        X_EX2_train, X_EX2_test, Y_EX2_train, Y_EX2_test = train_test_split(X_EX2, Y_EX2, test_size=0.3, random_state=0)#random_state = ?

        Train_counter = Counter(Y_EX2_train)
        Test_counter = Counter(Y_EX2_test)

        print('\n','モダンなリビングルーム　の学習用データ数：',Train_counter[0],'    ','モダンでないリビングルーム　の学習用データ数：',Train_counter[1], flush=True)
        print('モダンなリビングルーム　のテスト用データ数：',Test_counter[0],'    ','モダンでないリビングルーム　のテスト用データ数：',Test_counter[1],'\n', flush=True)"""

        using_xgboost_Grid_Search(X_EX2, Y_EX2, save_path, "modern_not_modern_living")
        
        print('--------------------------------------------------------------------------', flush=True)


    if cute_room_flag:
        # かわいい部屋　かわいくない部屋
        print('--------------------------------------------------------------------------', flush=True)
        # データを用意
        cute_room, cute_room_label = load_data(data_path + 'cute_room_result/img_dataset.csv',0)
        cute_room = cute_room * 100
        not_cute_room, not_cute_room_label = load_data(data_path + 'living_modern_result/img_dataset.csv', 1)
        not_cute_room = not_cute_room * 100

        #X (行＝scene番号，列＝特徴量)　行方向に連結
        X_EX3 = np.vstack((cute_room, not_cute_room))
        #Y　列方向に連結　1列
        Y_EX3 = np.hstack((cute_room_label, not_cute_room_label)) 

        # int型に変換    （nan = -2147483648）
        X_EX3 = X_EX3.astype(float)
        Y_EX3 = Y_EX3.astype(float) #int型をやめてfloat型に変更

        """# 学習データとテストデータに分割
        X_EX3_train, X_EX3_test, Y_EX3_train, Y_EX3_test = train_test_split(X_EX3, Y_EX3, test_size=0.3, random_state=0)#random_state = ?

        Train_counter = Counter(Y_EX3_train)
        Test_counter = Counter(Y_EX3_test)

        print('\n','かわいい部屋　の学習用データ数：',Train_counter[0],'    ','かわいくない部屋　の学習用データ数：',Train_counter[1], flush=True)
        print('かわいい部屋　のテスト用データ数：',Test_counter[0],'    ','かわいくない部屋　のテスト用データ数：',Test_counter[1],'\n', flush=True)"""

        using_xgboost_Grid_Search(X_EX3, Y_EX3, save_path, "cute_not_cute_room")
        
        print('--------------------------------------------------------------------------', flush=True)





if __name__ == "__main__":
    create_model(modern_office_flag=False, modern_living_flag=False ,cute_room_flag=True)