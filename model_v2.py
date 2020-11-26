import numpy as np
import datetime
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from data_preprocessing_v2 import Data_Preprocessing

data_processor = Data_Preprocessing()
def model_and_split(n_splits,seed, test_size):

    split_types = {"kfold"                    : KFold(n_splits=n_splits, shuffle=False, random_state=seed),
                   "stratifiedsuffleSplit" : StratifiedShuffleSplit(n_splits=n_splits, test_size = test_size ,random_state=seed),
                   "stratifiedKFold"      : StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=seed)
                  }

    models = {    "xgboost" : xgb.XGBClassifier(n_estimators=1000,
                                                 max_depth=6,
                                                 learning_rate=0.04,
                                                 subsample=0.9,
                                                 colsample_bytree=0.35,
                                                 objective = 'binary:logistic',
                                                 random_state = 1
                                                 ),

                  "catboost" : CatBoostClassifier(iterations=1000,
                                                  learning_rate=0.02,
                                                  random_strength=0.1,
                                                  depth=8,
                                                  loss_function='Logloss',
                                                  eval_metric='Logloss',
                                                  leaf_estimation_method='Newton',
                                                  random_state = 1,
                                                  subsample = 0.9,
                                                  rsm = 0.8
                                                  ),
                   "lgb" : lgb.LGBMClassifier(boosting_type='gbdt',
                                              n_estimators=10000,
                                              max_depth=10,
                                              learning_rate=0.02,
                                              subsample=0.9,
                                              colsample_bytree=0.4,
                                              objective ='binary',
                                              random_state = 1,
                                              importance_type='gain',
                                              reg_alpha=2,
                                              reg_lambda=2
                                              #cat_features=cat_features
                                             )                                    
                 }
    return split_types, models


def model_fit(X, Y, X_test, n_splits, split_type, model):
    '''
    X : Input Feature
    Y: label
    X_test : Test input Feature
    split_type : 1. KFold  ---> key:-kfold
                 2. StratifiedShuffleSplit ----> key:-stratifiedsuffleSplit
    model : Model name that you wnt to use
            i.e xgboost, catboost
    '''
    test_size = 0.25
    X_train_cv = X.copy()
    y_train_cv = Y.copy()
    probs = np.zeros(shape=(len(X_test),))
    scores = []
    avg_loss = []
    seed = 1
    probs_oof_train = np.zeros(shape=(len(y_train_cv),))
    print('#'*100)
    print("** "+ split_type + " **")
    print("** "+ model + " **")
    save_model = "models/" + model + "/" + split_type + "/"
    validate_dirs(save_model)
    split_types, models = model_and_split(n_splits,seed, test_size)
    spliter = split_types[split_type]

    for i, (idxT, idxV) in enumerate(spliter.split(X_train_cv, y_train_cv)):

        print('Fold', i)

        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))
        

        clf = models[model]

        if i==0:
            model_clf= clf

        if model=='catboost':
            h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT],
                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                   early_stopping_rounds=50,verbose = 100)
        elif model=='xgboost':
            h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 
                        eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                        verbose=100,eval_metric=['auc','logloss'],
                        early_stopping_rounds=50)
        elif model=='lgb':
            h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 
                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                    verbose=100,eval_metric=['binary_logloss','auc'],
                    early_stopping_rounds=100)                
        
        filename = save_model + model + "_" + split_type + '_Fold -{}-' +  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.sav'
        filename = filename.format(i)
        pickle.dump(h,open(filename,'wb'))

        probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]
        
        probs_oof_train  += clf.predict_proba(X_train_cv)[:,1]

        probs +=clf.predict_proba(X_test)[:,1]

        roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)

        scores.append(roc)

        print (model + ' Val OF AUC=',roc)
        
        if model=='catboost':
            avg_loss.append(clf.best_score_['validation']['Logloss'])
        elif model=='xgboost':
            avg_loss.append(clf.best_score)
        elif model=='lgb':
            avg_loss.append(clf.best_score_['valid_0']['binary_logloss'])    


        print('#'*100)


    print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))
    
    return probs, probs_oof_train, model_clf

def model_blending(xgboost, catboost, y_train):
    scores = []
    weights = []
    for w in range(0, 10):
        w = w/10
        result = w * xgboost + abs(1-w) * catboost
        roc = roc_auc_score(y_train, result)
        scores.append(roc)
        weights.append(w)
    plot_graph(scores, weights)
    max_roc = max(scores)
    max_roc_index = scores.index(max_roc)
    best_w = weights[max_roc_index]
    return best_w, max_roc

def plot_graph(x, y):
    plt.plot(x, y)
    plt.xlabel("Scores")
    plt.ylabel("weights")
    plt.title("roc_auc_score vs weights")
    plt.show()

def feature_importance(model, X_train, model_name):

    fI = model.feature_importances_
    print(fI)
    names = X_train.columns.values
    ticks = [i for i in range(len(names))]
    plt.bar(ticks, fI)
    plt.xticks(ticks, names,rotation = 90)
    plt.title(model_name)
    plt.xlabel("Features")
    plt.show()

def validate_dirs(dir):
    try: 
        if not os.path.exists(dir):
            os.makedirs(dir)  
    except OSError:
        print('Error: Creating directory to store person')

def model_inference(X, model_prefix, data_type, drop_columns, target_column, key,  prediction=None, predict_probs=None):
    '''
    X              : X is raw data
    model_prefix   : model_path
    data_type      : Train or Test
    drop_columns   : list of columns
    target_columns : label
    key            : is used for merge dataframe
    '''  
    import pickle, joblib
    import glob
    assert data_type == "Train" or data_type=="Test", "data_type must : Train or Test " 

    if data_type == "Train" :
        X, Y = data_processor.data_processing_pipeline(
            X, drop_columns , target_column, key, data_type = data_type) 
    elif data_type == "Test" :
        X = data_processor.data_processing_pipeline(
            X, drop_columns , target_column, key, data_type = data_type) 

    pridict = np.zeros(shape=(len(X),))
    models_path = glob.glob(model_prefix)
    for i, v in enumerate(models_path):
        print(i)
        model = joblib.load(v)
        if prediction:
            result = model.predict(X)
        if predict_probs:
            result = model.predict_proba(X)    
        pridict += result
    return pridict    

        
