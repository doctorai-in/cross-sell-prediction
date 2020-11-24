#####################################################################################
#                          XGBClassifier with StratifiedShuffleSplit                #
#####################################################################################
import numpy as np
import datetime
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.metrics import roc_auc_score


def simple_model_v1_xgboost(X, Y, X_test):
    '''
    X : Input Feature
    Y: label
    X_test : Test input Feature
    '''
    X_train_cv = X.copy()
    y_train_cv = Y.copy()

    clf = xgb.XGBClassifier(n_estimators=1000,
                                    max_depth=6,
                                    learning_rate=0.04,
                                    subsample=0.9,
                                    colsample_bytree=0.35,
                                    objective = 'binary:logistic',
                                    random_state = 1
                            )        

    h = clf.fit(X_train_cv, y_train_cv,
                        verbose=100,eval_metric=['auc','logloss'])

    result_simple_xgboost_deafault = clf.predict_proba(X_test)[:,1]

    filename = 'models/XGBoost_Simple-default-' +  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.sav' 
    pickle.dump(h,open(filename,'wb'))
    
    return result_simple_xgboost_deafault


def model_v2_xgboost_StratifiedSplit(X, Y, X_test):
    '''
    X : Input Feature
    Y: label
    X_test : Test input Feature
    '''
    test_size = 0.34
    X_train_cv = X.copy()
    y_train_cv = Y.copy()
    scores = []
    avg_loss = []
    seed = 1
    probs_xgb = np.zeros(shape=(len(X_test),))
    print('#'*100)
    sssf = StratifiedShuffleSplit(n_splits=5, test_size = test_size ,random_state=seed)

    for i, (idxT, idxV) in enumerate(sssf.split(X_train_cv, y_train_cv)):

            print('Fold',i)

            print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))

            clf = xgb.XGBClassifier(n_estimators=1000,
                                    max_depth=6,
                                    learning_rate=0.04,
                                    subsample=0.9,
                                    colsample_bytree=0.35,
                                    objective = 'binary:logistic',
                                    random_state = 1
                                   )        


            h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT], 
                        eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                        verbose=100,eval_metric=['auc','logloss'],
                        early_stopping_rounds=50)
            filename = 'models/XGBoost_Fold -{}-Simple-default-' +  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.sav'
            filename = filename.format(i)
            pickle.dump(h,open(filename,'wb'))

            probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]

            probs_xgb +=clf.predict_proba(X_test)[:,1]

            roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)

            scores.append(roc)

            avg_loss.append(clf.best_score)

            print ('XGB Val OOF AUC = ',roc)

            print('#'*100)
    print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))

    print('%.8f (%.8f)' % (np.array(scores).mean(), np.array(scores).std())) 
    
    return probs_xgb

#####################################################################################
#                          CatBoostClassifier with StratifiedShuffleSplit           #
#####################################################################################

def model_v3_catboost_StratifiedSplit(X, Y, X_test):
    '''
    X : Input Feature
    Y: label
    X_test : Test input Feature
    '''
    test_size = 0.34
    X_train_cv = X.copy()
    y_train_cv = Y.copy()
    probs_cb = np.zeros(shape=(len(X_test),))
    scores = []
    avg_loss = []
    seed = 1
    print('#'*100)
    sssf = StratifiedShuffleSplit(n_splits=5, test_size = test_size ,random_state=seed)

    for i, (idxT, idxV) in enumerate(sssf.split(X_train_cv, y_train_cv)):

        print('Fold', i)

        print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))

        clf = CatBoostClassifier(iterations=10000,
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

        h = clf.fit(X_train_cv.iloc[idxT], y_train_cv.iloc[idxT],
                    eval_set=[(X_train_cv.iloc[idxV],y_train_cv.iloc[idxV])],
                   early_stopping_rounds=50,verbose = 100)
        filename = 'models/CatBoost_Fold -{}-Simple-default-' +  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.sav'
        filename = filename.format(i)
        pickle.dump(h,open(filename,'wb'))

        probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]

        probs_cb +=clf.predict_proba(X_test)[:,1]

        roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)

        scores.append(roc)

        print ('CatBoost Val OOF AUC=',roc)

        avg_loss.append(clf.best_score_['validation']['Logloss'])


        print('#'*100)

    print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))
    
    return probs_cb

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
    test_size = 0.34
    X_train_cv = X.copy()
    y_train_cv = Y.copy()
    probs_cb = np.zeros(shape=(len(X_test),))
    scores = []
    avg_loss = []
    seed = 1
    models = {"xgboost" : xgb.XGBClassifier(n_estimators=1000,
                                             max_depth=6,
                                             learning_rate=0.04,
                                             subsample=0.9,
                                             colsample_bytree=0.35,
                                             objective = 'binary:logistic',
                                             random_state = 1
                                             ),
              
              "catboost" : CatBoostClassifier(iterations=10000,
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
    print('#'*100)
    split_types = {"kfold"                    : KFold(n_splits=n_splits, shuffle=False, random_state=seed),
                  "stratifiedsuffleSplit" : StratifiedShuffleSplit(n_splits=n_splits, test_size = test_size ,random_state=seed)
                  }
    print("** "+ split_type + " **")
    print("** "+ model + " **")
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
        
        filename = 'models/' + model + '_Fold -{}-Simple-default-' +  datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.sav'
        filename = filename.format(i)
        pickle.dump(h,open(filename,'wb'))

        probs_oof = clf.predict_proba(X_train_cv.iloc[idxV])[:,1]

        probs_cb +=clf.predict_proba(X_test)[:,1]

        roc = roc_auc_score(y_train_cv.iloc[idxV],probs_oof)

        scores.append(roc)

        print (model + ' Val OOF AUC=',roc)
        
        if model=='catboost':
            avg_loss.append(clf.best_score_['validation']['Logloss'])
        else:
            avg_loss.append(clf.best_score)


        print('#'*100)

    print("Log Loss Stats {0:.8f},{1:.8f}".format(np.array(avg_loss).mean(), np.array(avg_loss).std()))
    
    return probs_cb
