import numpy as np
import datetime
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
def model_and_split(n_splits,seed, test_size):

    split_types = {"kfold"                    : KFold(n_splits=n_splits, shuffle=False, random_state=seed),
                   "stratifiedsuffleSplit" : StratifiedShuffleSplit(n_splits=n_splits, test_size = test_size ,random_state=seed),
                   "stratifiedKFold"      : StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=seed)
                   }

    models = {"xgboost" : xgb.XGBClassifier(n_estimators=1000,
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
    split_types, models = model_and_split(n_splits,seed, test_size)
    spliter = split_types[split_type]

    for i, (idxT, idxV) in enumerate(spliter.split(X_train_cv, y_train_cv)):

        print('Fold', i)

        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))

        clf = models[model]

        if model=='catboost':
            h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT],
                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                   early_stopping_rounds=50,verbose = 100)
        else:
            h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 
                        eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                        verbose=100,eval_metric=['auc','logloss'],
                        early_stopping_rounds=50)
        
        filename = 'models/' + model + "_" + split_type + '_Fold -{}-' +  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.sav'
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
        else:
            avg_loss.append(clf.best_score)


        print('#'*100)

    print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))
    
    return probs, probs_oof_train

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
    plt.show()

def feature_importance(model, X_train):

    fI = model.feature_importances_
    
    print(fI)
    
    names = X_train.columns.values
    
    ticks = [i for i in range(len(names))]
    
    plt.bar(ticks, fI)
    
    plt.xticks(ticks, names,rotation = 90)
    
    plt.show()
        
        
