import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from scipy import stats
import time
import xgboost as xgb
from sklearn.metrics import roc_auc_score,mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
import sklearn.preprocessing as preprocessing
data_train=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\二手车交易价格预测\used_car_train_20200313.csv',sep=' '))
data_test=pd.DataFrame(pd.read_csv(r'C:\Users\朝花夕拾\Desktop\机器学习\tianchi\二手车交易价格预测\used_car_testB_20200421.csv',sep=' '))
data_train['type']='train'
data_test['type']='test'
'''print(data_train.info(verbose=True,null_counts=True))
print(data_test.info(verbose=True,null_counts=True))
print(data_train.head())
print(data_test.head())
print(data_train.shape)
print(data_test.shape)'''
data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True,sort=False)
#temp=pd.DataFrame(data_all.head())
#print(data_all.info())
#查看缺失值
missing=data_all.isnull().sum()
missing=missing[missing>0]
'''print(missing)
print(data_all['bodyType'].value_counts())
print(data_all['fuelType'].value_counts())
print(data_all['gearbox'].value_counts())'''
#print(data_all['model'].value_counts())

# 数值特征的数据分析
num_features=[ 'kilometer','power','price', 'v_0',
       'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
       'v_11', 'v_12', 'v_13', 'v_14']
categorical_features=['model','name', 'brand', 'notRepairedDamage','bodyType', 'fuelType', 'gearbox','regionCode']

'''for num in num_features:
    print('{}特征有{}个不同值'.format(num,data_all[num].nunique()))
    temp=data_all[num].value_counts()
    print(temp)
    print(pd.DataFrame(data_all[num]).describe())'''

# 分类特征的数据分析
'''for cat in categorical_features:
    print('{}特征有{}个不同值'.format(cat,data_all[cat].nunique()))
    temp=data_all[cat].value_counts()
    print(temp)
    print(pd.DataFrame(data_all[cat]).describe())'''
data_all['notRepairedDamage'].replace('-', np.nan, inplace=True)

# 删去不明显的特征
data_all=data_all.drop(['seller','offerType'],axis=1)
'''for cat in categorical_features:
    data_all[(data_all['type']=='train')][cat].value_counts().plot(kind='box')
    plt.show()'''

'''for num in num_features:
   sns.distplot(data_train[num],kde=False,fit=stats.norm)
   plt.show()'''

# train和test的特征分布情况
#print(data_all.columns)
feature=[ 'name',  'model', 'brand', 'bodyType', 'fuelType',
       'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode',
        'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6','v_7', 'v_8', 'v_9',
        'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
'''for i in feature:
    g=sns.kdeplot(data=data_all[i][(data_all['type']=='train')],color='Red',shade=True)
    g = sns.kdeplot(data=data_all[i][(data_all['type'] == 'test')],ax=g, color='Blue', shade=True)
    g.set_xlabel(i)
    g.set_ylabel("Frequency")
    g = g.legend(["train", "test"])
    plt.show()'''

# 分析特征的相关性
'''feature=['price','name',  'model', 'brand', 'bodyType', 'fuelType',
       'gearbox', 'power', 'kilometer', 'notRepairedDamage', 'regionCode',
        'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6','v_7', 'v_8', 'v_9',
        'v_10', 'v_11', 'v_12', 'v_13', 'v_14']
corr=data_all[feature].corr(method='spearman')
corr=pd.DataFrame(corr)
#print(corr)
sns.heatmap(corr,fmt='0.2f')
plt.show()'''

# 删掉相关性比较低的特征
data_all=data_all.drop(['regionCode'],axis=1)

# 时间特征
#print(data_all['regDate'].dtype)
data_all['regDate']= pd.to_datetime(data_all['regDate'], format='%Y%m%d', errors='coerce')
data_all['creatDate']=pd.to_datetime(data_all['creatDate'], format='%Y%m%d', errors='coerce')
#print(data_all)
#print(data_all['regDate'].isnull().sum()) # 由于在regDate里，有一些数据的格式出错，如20070009，可是我们没有0月！
#print(data_all['creatDate'].isnull().sum())
data_all['used_time']=(data_all['creatDate']-data_all['regDate']).dt.days
#print(data_all)

# 品牌和价格特征
brand_and_price_mean=data_all.groupby('brand')['price'].mean()
model_and_price_mean=data_all.groupby('model')['price'].mean()
brand_and_price_median=data_all.groupby('brand')['price'].median()
model_and_price_median=data_all.groupby('model')['price'].median()
data_all['brand_and_price_mean']=data_all.loc[:,'brand'].map(brand_and_price_mean)
data_all['model_and_price_mean']=data_all.loc[:,'model'].map(model_and_price_mean).fillna(model_and_price_mean.mean())
data_all['brand_and_price_median']=data_all.loc[:,'brand'].map(brand_and_price_mean)
data_all['model_and_price_median']=data_all.loc[:,'model'].map(model_and_price_mean).fillna(model_and_price_median.mean())


data_all=data_all.drop(['SaleID','regDate','creatDate','type'],axis=1)
# 处理缺失值
print(data_all.isnull().sum())
data_all['model']=data_all['model'].fillna(data_all['model'].median())
# 处理bodyType
X_bT=data_all.loc[(data_all['bodyType'].notnull()),:]
X_bT=X_bT.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
ybT_train=data_all.loc[data_all['bodyType'].notnull()==True,'bodyType']
print(X_bT.shape)
print(ybT_train.shape)
XbT_test=data_all.loc[data_all['bodyType'].isnull()==True,:]
XbT_test=XbT_test.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
print(XbT_test.shape)

# 处理fuelType
X_fT=data_all.loc[(data_all['fuelType'].notnull()),:]
X_fT=X_fT.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
yfT_train=data_all.loc[data_all['fuelType'].notnull()==True,'fuelType']
print(X_fT.shape)
print(yfT_train.shape)
XfT_test=data_all.loc[data_all['fuelType'].isnull()==True,:]
XfT_test=XfT_test.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
print(X_fT.shape)

#处理gearbox
X_gb=data_all.loc[(data_all['gearbox'].notnull()),:]
X_gb=X_gb.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
ygb_train=data_all.loc[data_all['gearbox'].notnull()==True,'gearbox']
print(X_gb.shape)
print(ygb_train.shape)
Xgb_test=data_all.loc[data_all['gearbox'].isnull()==True,:]
Xgb_test=Xgb_test.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
print(Xgb_test.shape)

#处理notRepairedDamage
X_nRD=data_all.loc[(data_all['notRepairedDamage'].notnull()),:]
X_nRD=X_nRD.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
ynRD_train=pd.DataFrame(data_all.loc[data_all['notRepairedDamage'].notnull()==True,'notRepairedDamage']).astype('float64')
print(X_nRD.shape)
print(ynRD_train.shape)

XnRD_test=data_all.loc[data_all['notRepairedDamage'].isnull()==True,:]
XnRD_test=XnRD_test.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
print(XnRD_test.shape)
print(X_nRD.info())
print(ynRD_train.info())
print(XnRD_test.info())
#处理used_time
scaler=preprocessing.StandardScaler()
uesed_time=scaler.fit(np.array(data_all['used_time']).reshape(-1, 1))
data_all['used_time']=scaler.fit_transform(np.array(data_all['used_time']).reshape(-1, 1),uesed_time)

X_ut=data_all.loc[(data_all['used_time'].notnull()),:]
X_ut=X_ut.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
yut_train=data_all.loc[data_all['used_time'].notnull()==True,'used_time']
print(X_ut.shape)
print(yut_train.shape)
Xut_test=data_all.loc[data_all['used_time'].isnull()==True,:]
Xut_test=Xut_test.drop(['bodyType','fuelType','gearbox','notRepairedDamage','price','used_time'],axis=1)
print(Xut_test.shape)

# 用Xgboost填充缺失值
#print(X_bT.isnull().sum())
#print(XbT_test.isnull().sum())
def RFmodel(X_train,y_train,X_test):
    model_xgb= xgb.XGBRegressor(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2)
    model_xgb.fit(X_train,y_train)
    y_pre=model_xgb.predict(X_test)
    return y_pre
y_pred=RFmodel(X_bT,ybT_train,XbT_test)
data_all.loc[data_all['bodyType'].isnull(),'bodyType']=y_pred

y_pred0=RFmodel(X_fT,yfT_train,XfT_test)
data_all.loc[data_all['fuelType'].isnull(),'fuelType']=y_pred0

y_pred1=RFmodel(X_gb,ygb_train,Xgb_test)
data_all.loc[data_all['gearbox'].isnull(),'gearbox']=y_pred1

y_pred2=RFmodel(X_nRD,ynRD_train,XnRD_test)
data_all.loc[data_all['notRepairedDamage'].isnull(),'notRepairedDamage']=y_pred2


y_pred3=RFmodel(X_ut,yut_train,Xut_test)
data_all.loc[data_all['used_time'].isnull(),'used_time']=y_pred3


#print(data_all)
    #score=cross_val_score(model_Rf,X_bT,ybT_train)
    #print(score.mean())

#模型
X=data_all.loc[0:149999,:]
X=X.drop(['price'],axis=1)
Y=data_all.loc[0:149999,'price']

X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=0)
print(",训练数据特征:",X_train.shape,
      ",测试数据特征:",X_test.shape)
print(",训练数据标签:",y_train.shape,
     ',测试数据标签:',y_test.shape)
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
model_xgb=xgb.XGBClassifier(max_depth=4, colsample_btree=0.1, learning_rate=0.1, n_estimators=32, min_child_weight=2)
model_xgb.fit(X_train.astype('float64'),y_train)
Y_pred=model_xgb.predict(X_test.astype('float64'))
score=mean_absolute_error(y_test,Y_pred)
print(score)


#Y_test=data_all.loc[150000:,:]
#Y_test=Y_test.drop(['price'])
#Y_pred=model_xgb.predict(Y_test)
