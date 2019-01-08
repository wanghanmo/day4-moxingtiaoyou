# day4-模型调优
## 实验要求 
使用网格搜索法对7个模型进行调优（调参时采用五折交叉验证的方式），并进行模型评估。
## 代码
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def evaluate(pre,pre_pro,y):
    acc=accuracy_score(y,pre)
    model_auc=roc_auc_score(y,pre)
    model_pre=precision_score(y,pre)
    model_recall=recall_score(y,pre)
    model_f1=f1_score(y,pre)
    fpr, tpr, thresholds =roc_curve(y,pre_pro[:,1])
    return acc,model_auc,model_pre,model_recall,model_f1,fpr,tpr

def model_plot(fpr,tpr,name):
    #plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % model_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s的ROC曲线'%name)
    plt.legend(loc="lower right")
    plt.savefig('%s的ROC曲线.png'%name)
    plt.show()


data=pd.read_csv('data_all.csv')

y=data['status']
x=data.drop(['status'],axis=1)
scaler=StandardScaler()
X=scaler.fit_transform(x)
print('the size of X,y:',X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)



params_lr = {'solver': ['newton-cg', 'lbfgs', 'sag'], 'C': [0.1, 1, 10]}
lr_model = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), params_lr, cv=5, scoring='roc_auc')

params_svm = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10]}
svm_model = GridSearchCV(SVC(class_weight='balanced', gamma='auto', probability=True), params_svm, cv=5, scoring='roc_auc')

params_dt = {'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2', None]}
dt_model = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), params_dt, cv=5, scoring='roc_auc')

params_rf = {'n_estimators': range(10, 80, 10), 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2', None]}
rf_model = GridSearchCV(RandomForestClassifier(class_weight='balanced'), params_rf, cv=5, scoring='roc_auc')

params_en = {'n_estimators': range(10, 80, 10)}
gbdt_model = GridSearchCV(GradientBoostingClassifier(), params_en, cv=5, scoring='roc_auc')
xg_model = GridSearchCV(XGBClassifier(), params_en, cv=5, scoring='roc_auc')
lgb_model = GridSearchCV(LGBMClassifier(), params_en, cv=5, scoring='roc_auc')


models=[('LogisticRegression',lr_model),
        ('SVM',svm_model),
        ('DecisionTreeClassifier',dt_model),
        ('RandomForestClassifier',rf_model),
        ('GradientBoostingClassifier',gbdt_model),
        ('XGBClassifier',xg_model),
        ('LGBMClassifier',lgb_model)]
        
for name,model in models:
    print(name,'Start training...')
    model.fit(X_train,y_train)
    print(name,model.best_params_)
    preds=model.predict(X_test)
    pro=model.predict_proba(X_test)
    acc,model_auc,model_pre,model_recall,model_f1,fpr,tpr=evaluate(preds,pro,y_test)
    print(name,'accuracy_score,roc_auc_score,precision_score,recall_score,f1_score：',acc,model_auc,model_pre,model_recall,model_f1)
    model_plot(fpr,tpr,name)
```
### 最佳参数
各个模型通过网格搜索的参数如下：
```python
LogisticRegression {'C': 10, 'solver': 'sag'}
SVM {'C': 1, 'kernel': 'linear'}
DecisionTreeClassifier {'criterion': 'entropy', 'max_features': None}
RandomForestClassifier {'criterion': 'entropy', 'max_features': None, 'n_estimators': 70}
GradientBoostingClassifier {'n_estimators': 40}
XGBClassifier {'n_estimators': 50}
LGBMClassifier {'n_estimators': 20}
```
## 实验结果
## 结果
模型|准确率|精确率|召回率|F1-score|AUC值|ROC曲线
---|------|-----|-----|-----|-----|------
LogisticRegression|0.7126839523475823|0.6850815832576967|0.44930417495029823|0.6295264623955432|0.5243619489559165|![LogisticRegression](https://github.com/wanghanmo/day3_-/blob/master/LogisticRegression%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)    
SVM|0.7035739313244569|0.6799200338017589|0.43822393822393824|0.6323119777158774|0.5176738882554163|![SVM](https://github.com/wanghanmo/day3_-/blob/master/SVM%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)
DecisionTreeClassifier|0.6895585143658024|0.5984384943611571|0.3900523560209424|0.415041782729805|0.40215924426450744|![DecisionTreeClassifier](https://github.com/wanghanmo/day3_-/blob/master/DecisionTreeClassifier%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)
RandomForestClassifier|0.7792571829011913 |0.6195306876154111 |0.6294117647058823 |0.298050139275766 |0.4045368620037807|![RandomForestClassifier](https://github.com/wanghanmo/day3_-/blob/master/RandomForestClassifier%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)
GradientBoostingClassifier|0.7883672039243167 |0.6376365371975838 |0.6557377049180327| 0.3342618384401114| 0.44280442804428044|![GradientBoostingClassifier](https://github.com/wanghanmo/day3_-/blob/master/GradientBoostingClassifier%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)
XGBClassifier| 0.7855641205325858 |0.6339146922892345| 0.644808743169399 |0.3286908077994429 |0.4354243542435425|![XGBClassifier](https://github.com/wanghanmo/day3_-/blob/master/XGBClassifier%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)
LGBMClassifier|0.7729502452697968 |0.6106942401385455| 0.6035502958579881 |0.2841225626740947 |0.3863636363636364|![LGBMClassifier](https://github.com/wanghanmo/day3_-/blob/master/LGBMClassifier%E7%9A%84ROC%E6%9B%B2%E7%BA%BF.png)
## 参考
- sklearn官方英文文档：https://scikit-learn.org/stable/index.html
- sklearn中文版文档：http://sklearn.apachecn.org/#/
- xgboost官方英文文档：https://xgboost.readthedocs.io/en/latest/
- lightgbm英文官方文档：https://lightgbm.readthedocs.io/en/latest/
