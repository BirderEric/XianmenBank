import pandas as pd
import numpy as np
import warnings
import math
import os
import lightgbm as lgb
from  sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score,cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

def gen_pesudo_data(): 
    tmp_aum = pd.read_csv('./train/aum/aum_m10.csv')
    tmp_aum[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']] = np.nan
    tmp_aum.to_csv('./train/aum/aum_m6.csv',index=None)
    tmp_cunkuan = pd.read_csv('./train/cunkuan/cunkuan_m10.csv')
    tmp_cunkuan[['C1', 'C2']] = np.nan
    tmp_cunkuan.to_csv('./train/cunkuan/cunkuan_m6.csv',index=None)
    tmp_beh = pd.read_csv('./train/behavior/behavior_m9.csv')
    tmp_beh[['B1', 'B2','B3', 'B4','B5', 'B6','B7']] = np.nan
    tmp_beh.to_csv('./train/behavior/behavior_m6.csv',index=None)
    
class GetData(object):

    def load_aum_data(self,path='./',m=1,iftrain=True):  
        data = pd.read_csv(path + 'aum_m{}.csv'.format(m) )
        m =  m if iftrain else m+12
        data.columns = ['cust_no', 'X1_{}'.format(m), 'X2_{}'.format(m), 'X3_{}'.format(m), 'X4_{}'.format(m),
                      'X5_{}'.format(m), 'X6_{}'.format(m), 'X7_{}'.format(m), 'X8_{}'.format(m)]
        return data
    
    def load_cunkuan_data(self,path='./',m=1,iftrain=True):
        data = pd.read_csv(path + 'cunkuan_m{}.csv'.format(m) )
        data['C3'] = data['C1'] / (1+data['C2'])
        m =  m if iftrain else m+12
        data.columns = ['cust_no', 'C1_{}'.format(m), 'C2_{}'.format(m), 'C3_{}'.format(m)]
        return data
    
    def load_beh_data(self,path='./',m=1,iftrain=True):
        data = pd.read_csv(path + 'behavior_m{}.csv'.format(m) )
        m =  m if iftrain else m+12
        if m in [3,6,9,12,15]:
            data['x'] = pd.to_datetime(data['B6']) + pd.tseries.offsets.QuarterEnd() 
            data['B6'] = (pd.to_datetime(data['x']) - pd.to_datetime(data['B6'])).dt.days # 该季度与最后一次交易时间距离的天数
            data = data.drop('x',axis=1)
            data.columns = ['cust_no', 'B1_{}'.format(m), 'B2_{}'.format(m), 'B3_{}'.format(m), 'B4_{}'.format(m),
                      'B5_{}'.format(m), 'B6_{}'.format(m),'B7_{}'.format(m)]
        else:
            data['B6'] = np.nan
            data['B7'] = np.nan
            data.columns = ['cust_no', 'B1_{}'.format(m), 'B2_{}'.format(m), 'B3_{}'.format(m), 'B4_{}'.format(m),
                      'B5_{}'.format(m),'B6_{}'.format(m),'B7_{}'.format(m)]
        return data
def get_data(data_file='',param=''):
    df_aum = pd.DataFrame()
    for mon in range(3,0,-1):
        df = getattr(GetData(), param)(path='./test/{}/'.format(data_file),m= mon,iftrain=False)
        if len(df_aum):
            df_aum = df_aum.merge(df,on='cust_no',how='left')
        else:
            df_aum = df
    for mon in range(12,5,-1):
        df = getattr(GetData(), param)(path='./train/{}/'.format(data_file),m= mon,iftrain=True)
        df_aum = df_aum.merge(df,on='cust_no',how='left')
    return df_aum

def get_quater_data(df,data_type = '',colhead='X',num_col=3,quater=3):
    if data_type == 'aum':
        columns = ['cust_no']
        rename_col = ['cust_no']
        for i in range(1,num_col + 1):
            for j in range(4):
                columns.append('{}{}_{}'.format(colhead,i,quater*3-j))
                rename_col.append('{}{}_{}'.format(colhead,i,4 - j))
        tmp = df[columns]
        tmp.columns = rename_col
        return tmp
def statistics_feature_aum(df):

    for i in range(1,9):
        df['X{}_mean'.format(i)] = df[['X{}_4'.format(i),'X{}_3'.format(i),'X{}_2'.format(i)]].mean(axis=1)
        df['X{}_std'.format(i)] = df[['X{}_4'.format(i),'X{}_3'.format(i),'X{}_2'.format(i)]].std(axis=1)
        df['X{}_sum'.format(i)] = df[['X{}_4'.format(i),'X{}_3'.format(i),'X{}_2'.format(i)]].sum(axis=1)
        df['X{}_max'.format(i)] = df[['X{}_4'.format(i),'X{}_3'.format(i),'X{}_2'.format(i)]].max(axis=1)
        df['X{}_min'.format(i)] = df[['X{}_4'.format(i),'X{}_3'.format(i),'X{}_2'.format(i)]].min(axis=1)
    return df

def aum_feat_engineering(df):
    df['X3_sub1'] = df['X3_4']  - df['X3_3']
    df['X3_sub2'] = df['X3_4']  - df['X3_2']
    df['X3_sub3'] = df['X3_3']  - df['X3_2']
    df['X3_sub4'] = df['X3_2']  - df['X3_1']
    df['X3_sub5'] = df['X3_3']  - df['X3_1']

    df['X1_sub1'] = df['X1_4']  - df['X1_3']
    df['X1_sub2'] = df['X1_4']  - df['X1_2']
    df['X1_sub3'] = df['X1_4']  - df['X1_1']
    df['X1_sub4'] = df['X1_4']  - df['X1_1']

    df['X7_sub1'] = df['X7_4']  - df['X7_3']
    df['X7_sub2'] = df['X7_4']  - df['X7_2']
    df['X7_sub3'] = df['X7_4']  - df['X7_1']

    df['sum_X123_4'] = df[['X1_4','X2_4','X3_4']].sum(axis=1)
    df['sum_X123_3'] = df[['X1_3','X2_3','X3_3']].sum(axis=1)
    df['sum_X123_2'] = df[['X1_2','X2_2','X3_2']].sum(axis=1)
    df['sum_X123_1'] = df[['X1_1','X2_1','X3_1']].sum(axis=1)

    df['X123_sub1'] = df['sum_X123_4']  - df['sum_X123_3']
    df['X123_sub2'] = df['sum_X123_4']  - df['sum_X123_2']
    df['X123_sub3'] = df['sum_X123_4']  - df['sum_X123_1']

    df['X7_div_X123_sum'] = df['X7_4'] / (1e-3 + df['sum_X123_4'])
    df['X7_sub_X123_sum'] = df['X7_4'] - df['sum_X123_4']
    return df

def eval_error(pred,train_set):
    
    labels = train_set.get_label()
    pred = pred.reshape((3,int(len(pred)/3))).T
    y_pred = pred.argmax(axis=1)
    score = cohen_kappa_score(labels,y_pred)
    return 'kappa_score',score,True


def model_train(df,trainlabel,cate_cols,test_,feature,num_class):
    '''
    @param df: 训练数据 DataFrame
    @param trainlabel：训练标签 string  eg. 'label'
    @param cate_cols: 类别变量名 list  eg. ['col1','col2'...]
    @param test_ : 测试数据 DataFrame
    @param feature ：所有训练特征 list  eg. ['feat1','feat2'...]

    @return sub_preds: 预测数据
    
    '''
    train_= df.copy()
    auc = []
    n_splits = 5
    oof_lgb = np.zeros([len(train_),num_class])
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=2019)
    stratifiedKfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2019)
    sub_preds = np.zeros([test_.shape[0],num_class])
    sub_preds1 = np.zeros([test_.shape[0],n_splits])
    use_cart = True
    cate_cols = cate_cols
    label = trainlabel
    pred = list(feature)
    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective':'multiclass',
        #'metric':'multi-error',
        'num_class':num_class,
        'num_leaves':60,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 5,
        'min_data_in_leaf': 20,
        'max_depth':-1,
        'nthread': 8,
        'verbose': 1,
 #       'is_unbalanace':True,
#         'lambda_l1': 0.4,  
#         'lambda_l2': 0.5, 
        'device': 'gpu'
    }
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_[pred],train_[[label]]), start=1):
        print('the %s training start ...'%n_fold)

        train_x, train_y,train_weight = train_[pred].iloc[train_idx], train_[[label]].iloc[train_idx],train_[['weight']].iloc[train_idx]
        valid_x, valid_y,valid_weight = train_[pred].iloc[valid_idx], train_[[label]].iloc[valid_idx],train_[['weight']].iloc[valid_idx]
        #x = train_['bid'].iloc[valid_idx]
        print(train_y.shape)
        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols,weight=train_weight.values.flatten(order='F'))
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
            #dvalid1 = lgb.Dataset(test_[pred], label=test_[['label']], categorical_feature=cate_cols)

        else:
            dtrain = lgb.Dataset(train_x, label= train_y)
            dvalid = lgb.Dataset(valid_x, label= valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=1000,
            valid_sets=[dvalid],
           # early_stopping_rounds = 100,
            verbose_eval=100
           ,feval=eval_error
        )
        
        sub_preds += clf.predict(test_[pred].values,num_iteration=1000)/ folds.n_splits
        sub_preds1[:,n_fold-1] = clf.predict(test_[pred].values,num_iteration=400).argmax(axis=1)
        train_pred = clf.predict(valid_x,num_iteration=clf.best_iteration)
        y_pred = train_pred.argmax(axis=1)
        oof_lgb[valid_idx] = train_pred
    #print('MEAN AUC:',np.mean(auc))
    
    return sub_preds,oof_lgb,clf,sub_preds1

def main():
    
    df_aum = get_data(data_file='aum',param='load_aum_data')
    df_aum_test = get_quater_data(df_aum,data_type='aum',quater=5,colhead='X',num_col=8)[df_aum_columns]
    df_aum_test_Q4 = get_quater_data(df_aum,data_type='aum',quater=4,colhead='X',num_col=8)[df_aum_columns]
    df_aum_test_Q3 = get_quater_data(df_aum,data_type='aum',quater=3,colhead='X',num_col=8)[df_aum_columns]

    df_aum_test = aum_feat_engineering(df_aum_test)
    df_aum_test_Q4 = aum_feat_engineering(df_aum_test_Q4)
    df_aum_test_Q3 = aum_feat_engineering(df_aum_test_Q3)
    df_aum_test = statistics_feature_aum(df_aum_test)
    df_aum_test_Q4 = statistics_feature_aum(df_aum_test_Q4)
    df_aum_test_Q3 = statistics_feature_aum(df_aum_test_Q3)

    cust_avli_Q3 = pd.read_csv('./train/avli/cust_avli_Q3.csv')
    cust_avli_Q4 = pd.read_csv('./train/avli/cust_avli_Q4.csv')
    cust_avli_test = pd.read_csv('./test/avli/cust_avli_Q1.csv')

    cust_avli_Q3 = df_aum_test_Q3.loc[df_aum_test_Q3.cust_no.isin(cust_avli_Q3.cust_no.values)]
    cust_avli_Q4 = df_aum_test_Q4.loc[df_aum_test_Q4.cust_no.isin(cust_avli_Q4.cust_no.values)]
    cust_avli_test = df_aum_test.loc[df_aum_test.cust_no.isin(cust_avli_test.cust_no.values)]

    label_Q3 = pd.read_csv('./train/y/y_Q3_3.csv')
    label_Q4 = pd.read_csv('./train/y/y_Q4_3.csv')
    label_Q3['label'] = label_Q3.label + 1
    label_Q4['label'] = label_Q4.label + 1
    cust_avli_Q3 = cust_avli_Q3.merge(label_Q3,on='cust_no',how='left')
    cust_avli_Q4 = cust_avli_Q4.merge(label_Q4,on='cust_no',how='left')

    cust_avli_Q3['pre_label'] = np.nan
    last_label = label_Q3[['cust_no','label']]
    last_label.columns = ['cust_no','pre_label']
    cust_avli_Q4 = cust_avli_Q4.merge(last_label,on='cust_no',how='left')

    last_label = label_Q4[['cust_no','label']]
    last_label.columns = ['cust_no','pre_label']
    cust_avli_test = cust_avli_test.merge(last_label,on='cust_no',how='left')

    Train_data = pd.concat([cust_avli_Q3,cust_avli_Q4],axis=0)
    Train_data['weight'] = Train_data.label.map({1:1.03,2:0.58,0:1})
    feature = Train_data.drop(['cust_no','label','weight'],axis=1).columns
    sub_preds,oof_lgb,clf,sub_preds1 = model_train(Train_data,trainlabel='label',cate_cols=[],test_=cust_avli_test,feature=feature,num_class=3)
    cust_avli_test['label'] = sub_preds.argmax(axis=1) - 1
    cust_avli_test[['cust_no','label']].to_csv('./test/baseline.csv',index=None)

if __name__ == "__main__":
    main()