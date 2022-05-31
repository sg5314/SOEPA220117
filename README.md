# SOEPA220117
「知的オフィス環境推進協議会（SOEPA）1月定例研究会」でのデモアプリ

[知的オフィス環境推進協議会](http://mwind.jp/soepa/index.html)

# DEMO
* SOEPAでは，「家具推薦システム」と「テクスチャ調査システム」のデモを行いました．
    * 家具推薦システム：「家具」と「雰囲気」を選択することで，組み合わせの良い家具を推薦
    * テクスチャ調査システム：調査したい **「椅子」** を選択することで，テクスチャを調査
* 現在は，革調か木目調を判別することが可能


#### 家具推薦推薦システム
![SOEPA202201](https://user-images.githubusercontent.com/88835817/170429369-e2beb82d-e354-4fea-9911-7d9099258f4a.gif)

#### テクスチャ調査システム
![SOEPA202201_LBP](https://user-images.githubusercontent.com/88835817/171096507-c1166acf-b4eb-46f4-907c-ea74e777dbc7.gif)


# Installation & Usage

### 1.プロジェクトをクローン

    git clone https://github.com/sg5314/SOEPA220117.git

### 2.Dockerイメージの作成

    docker-compose build

### 3.Dockerコンテナの起動

    docker-compose up

### 4.以下のURLにアクセス

* http://127.0.0.1:3000/


## Dockerコンテナの削除

    docker-compose down

# Note

* 「/static/img/00_result_imgs」には，テクスチャ抽出に使用する送信された画像が保存．
    * ローカルで動かす際は，適度に中身を空にしてください．
    * Dockerコンテナで使う場合は，コンテナ削除で消えるので問題ないです．
　
* テクスチャ調査システムでは，[LBP特徴量](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html)を使用
 
# Author

* sg5314

