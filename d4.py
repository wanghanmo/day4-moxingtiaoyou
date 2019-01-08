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