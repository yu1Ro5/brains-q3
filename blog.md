## はじめに

先日公開したブログのQ3バージョンです．
参加したコンテストの概要については，こちらをご覧ください．
この記事では第5回富士フイルムBrain(s)コンテストのQ3に関する解法についてまとめていきます．

## ベストスコアモデル

* 特徴量は，1629
* モデルは，LightGBMRegressor
* パラメータは，あるタイミングでoptunaを使用してチューニングしたもの
* 最終出力は，5-folds CV の平均

## 特徴量生成

* SMILESからRDkitを用いて生成した200列の記述子
* SMILESの単純な文字列長(1列)
* SMLIESに含まれる元素のカウント(37列)
* MACCS Keys(167列)
* Morgan Fingerprint(1024列)
* [PolynomicalFeatures()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=poly#sklearn.preprocessing.PolynomialFeatures) を200列の記述子に適用して作成した2次の交差項20100列からLightGBMで特徴選択した200列

## モデル選択

