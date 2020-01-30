import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle


tv = pd.read_csv('tvfix.csv', sep=";")
tv.head()

def _get_category_mapping(column):
    """ Return the mapping of a category """
    return dict([(cat, code) for code, cat in enumerate(column.cat.categories)])

tv['produk'] = tv['produk'].astype('category')
produk_mapping = _get_category_mapping(tv['produk'])
tv['produk'] = tv['produk'].cat.codes

tv['merek'] = tv['merek'].astype('category')
merek_mapping = _get_category_mapping(tv['merek'])
tv['merek'] = tv['merek'].cat.codes

tv['tipe'] = tv['tipe'].astype('category')
tipe_mapping = _get_category_mapping(tv['tipe'])
tv['tipe'] = tv['tipe'].cat.codes

tv['ukuran'] = tv['ukuran'].astype('category')
ukuran_mapping = _get_category_mapping(tv['ukuran'])
tv['ukuran'] = tv['ukuran'].cat.codes

x = tv.drop(['harga'],axis=1)
y = tv['harga']
x.head()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=0)
classifier= RandomForestClassifier(n_estimators=100, max_features=4, random_state=0)
classifier.fit(x_train, y_train)


pickle.dump(classifier, open('model.pkl','wb'))
from sklearn.externals import joblib

def _save_variable(variable, filename):
    """ Save a variable to a file """
    joblib.dump(variable, filename)
    
_save_variable(produk_mapping, 'produk.pkl')
_save_variable(merek_mapping, 'merek.pkl')
_save_variable(tipe_mapping, 'tipe.pkl')
_save_variable(ukuran_mapping, 'ukuran.pkl')
