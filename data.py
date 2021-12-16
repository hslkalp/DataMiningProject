# Packages related to general operating system & warnings
from enum import auto
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
import statsmodels.tsa as tsa
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored as cl  # text customization
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import warnings
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree
from sklearn.datasets import load_iris
from os import system
import graphviz
warnings.filterwarnings('ignore')


# * importing database

data = pd.read_csv("creditcard.csv")

# * işlem sayısı
total_transactions = len(data)
normal = len(data[data.Class == 0])  # ! normal işlemler
fraudulent = len(data[data.Class == 1])  # ! dolandırılıcık işlemleri
fraud_percentage = round(fraudulent / normal * 100,
                         2)  # ! dolandırıcılık yüzdesi

print(cl('Toplam İşlem Sayısı {}'.format(
    total_transactions), attrs=['bold']))
print(cl('Normal İşlem Sayısı {}'.format(
    normal), attrs=['bold']))
print(cl('Dolandırılıcık işlem sayısı {}'.format(
    fraudulent), attrs=['bold']))
print(cl('Dolandırıcılık işlemlerinin yüzdesi {}'.format(
    fraud_percentage), attrs=['bold']))
# * null değerli veriler
# data.info()
# *miktar
# ! mininum ve maksimum tutarların arasındaki fark çok büyük olduğundan sonuç sapabilir
print("minimum ev maksimum harcama miktarı: ",
      min(data.Amount), max(data.Amount))

# * standart ölçeklendirmeyi kullanacağız
# * Veri standardizasyonu, öznitelikleri, ortalamaları 0 ve varyansı 1 olacak şekilde yeniden ölçeklendirme işlemidir.
#  * Standardizasyonu gerçekleştirmenin nihai amacı, değerlerin aralığındaki farklılıkları bozmadan tüm özellikleri ortak bir ölçeğe indirgemektir.
# * sklearn.preprocessing.StandardScaler() içinde, merkezleme ve ölçekleme her bir özellik üzerinde bağımsız olarak gerçekleşir.
# * Fit_transform(), eğitim verilerini ölçekleyebilmemiz ve ayrıca bu verilerin ölçekleme parametrelerini öğrenebilmemiz için eğitim verilerinde kullanılır. Burada, tarafımızca oluşturulan model, eğitim setinin özelliklerinin ortalamasını ve varyansını öğrenecektir. Bu öğrenilen parametreler daha sonra test verilerimizi ölçeklendirmek için kullanılır.
# * reshape fonksiyonu ile 1 boyutunda dizi oluşturuyoruz

sc = StandardScaler()
amount = data['Amount'].values
# * Eğitim verimizde fit_transform() yönetimi kullanıyoruz
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))


# * Zaman Stünunu kaldırıyoruz
data.drop(['Time'], axis=1, inplace=True)


print("veri setinin şekli: ", data.shape)


# * tekrarlanan işlemleri kaldıralım

data.drop_duplicates(inplace=True)
print("tekrarlanan işlemleri sildikten sonra veri setinin şekli: ", data.shape)

# * bağımlı bağımsız değişken

x = data.drop('Class', axis=1).values  # * bağımsız değişkendir
# * bağımlı değişken 0 ve 1 değerini almak için verileri ihtiyaç duyar x e bağımlıdır yani
y = data['Class'].values


# print("x: ", x)
# print("y: ", y)

# * modeli eğitmek için kullanacağımız veri seti ve test veri setini oluşturduk
# * Denetimli makine öğrenimi , verilen girdileri (bağımsız değişkenler veya tahmin ediciler ) verilen çıktılara (bağımlı değişkenler veya yanıtlar ) tam olarak eşleyen modeller oluşturmakla ilgilidir .
# * Anlaşılması gereken en önemli şey, bu ölçüleri doğru bir şekilde kullanmak, modelinizin tahmin performansını değerlendirmek ve modeli doğrulamak için genellikle tarafsız değerlendirmeye ihtiyaç duymanızdır .

# * Bu, eğitim için kullandığınız verilerle bir modelin tahmine dayalı performansını değerlendiremeyeceğiniz anlamına gelir. Modelin daha önce görmediği yeni verilerle modeli değerlendirmeniz gerekir . Bunu, kullanmadan önce veri kümenizi bölerek başarabilirsiniz.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=1)

# print("x ler: ", x_train, x_test)
# print("y ler: ", y_train, y_test)

# * Desicion Tree karar ağacı
# * Karar ağaçları yüksek boyuttaki verileri iyi işleyebilir.
dt = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# ! x,y eğitim kümesinden bir kara ağacı sınıflandırıcısı oluşturduk
dt.fit(x_train, y_train)
# ! ilişkiyi ölçmek için kullandığımız metod tahmin etmek
# ! Test veri kümesi için sonucu tahmin etmek
dt_pred = dt.predict(x_test)

# * Kara ağacı model doğruluğunu kontrol etme

# * Doğruluk, gerçek test seti değerleri ile tahmin edilen değerler karşılaştırılarak hesaplanabilir.
print(cl('Karar Ağacı Modelinin Doğruluk puanı {}'.format(
    metrics.accuracy_score(y_test, dt_pred)), attrs=['bold']))


print(cl('Kara Ağacının F1 puanı  {}'.format(
    f1_score(y_test, dt_pred)), attrs=['bold']))


# * Karışıklık Matrisini kontrol etme
print("Karışıklık matrisi: ", confusion_matrix(y_test, dt_pred, labels=[0, 1]))

text_representation = tree.export_text(dt)
print(text_representation)

feature_names = data.columns[:29]
target_names = ['0', '1']

# * Karar ağacını görselleştirme

fig = plt.figure(figsize=(25, 20), clear=True,)

tree.plot_tree(dt,
               feature_names=feature_names,
               class_names=target_names,
               filled=True,
               rounded=True)
fig.savefig('tree_visualization.png', bbox_inches='tight')
