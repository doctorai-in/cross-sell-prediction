from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from IPython.display import display
class Data_Preprocessing:
   
    def __init__(self):
        self.onehot_encoder = OneHotEncoder()
        self.label_encoder = LabelEncoder()
        self.values = None
    
    def _dataSplit_type(self, df, key, debug=True):
        '''
        Return categorical_df, numerical_df
        '''
        categorical_df = df.loc[:,df.dtypes==np.object]
        numerical_df = df.loc[:,df.dtypes!=np.object]
        if debug : 
            print("Categorical_Columns :: ", categorical_df.columns)
            print("Numerical_Columns :: ", numerical_df.columns)
        
        categorical_df[key]= df[key]
        return categorical_df, numerical_df
    
    def onehot_encoding(self, categorical_df, key, data_type, custom_encode_col=None):
        '''
        categorical_df : Dataframe which contains only categorical columns
        '''
        df = categorical_df.copy()
        columns  = list(df.columns)
        columns.remove(key)
        print(" ")
        print("## :: " + data_type + " :: ##")
        for col in columns:            
            print("CATEGORICAL COLUMN ENCODED :: ", col)
            df[col] = self.label_encoder.fit_transform(df[col])
        df = pd.get_dummies(df,prefix=columns, columns = columns)
            

    
        return df
    
    def drop_columns(self, df, columns):
        '''
        Columns : List of columns
        df      : Dataframe
        '''
        df = df.drop(columns, axis=1)
        
        return df
    
    def merge_df(self, categorical_df, numerical_df, key):
        '''
        merge two dataframe
        '''
        print(len(numerical_df))
        
        merged_df = pd.merge(numerical_df, categorical_df, on=key, how='left')
        
        return merged_df
    
    def corr_method_pandas(self, categorical, numerical):
        '''
        Find correlation using Pandas :
        '''
        corre = pd.concat([categorical,  numerical], axis=1, keys=['categorical', 'numerical']).corr().loc['numerical', 'categorical']
        display(corre)
        
    
    def corr_method_numpy(self, df1, df2):
        '''
        Find correlation using Numpy :
        '''
        n = len(df1)
        v1, v2 = df1.values, df2.values
        sums = np.multiply.outer(v2.sum(0), v1.sum(0))
        stds = np.multiply.outer(v2.std(0), v1.std(0))
        return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
                            df2.columns, df1.columns)    
    
    def data_processing_pipeline(self, df, drop_columns , target_column, key, data_type = 'Train', custom_encode_col=None):
        '''
        df : Dataframe
        drop_columns : List of columns
        label_column : label column name or target column name
        data_type : Either Train or Test. 
                    Use Train : if DATASET is full i.e with label  
                    Use Test:  if DATASET wihout label.
        key : key is column name through which we merge dataframe
        custom_encode_col : column name use for to encode with custom define encode values.
        
        RETURN 1. TRAIN ---> X,Y, final_df
               2. TEST  ---> X
        '''
        if data_type == 'Train':
            categorical_df, numerical_df = self._dataSplit_type(df, key)
            categorical_df               = self.onehot_encoding(categorical_df, key, data_type, custom_encode_col)
            
            print("")
            print(":: Pandas correlation : Categorical VS Numerical :: ")
            self.corr_method_pandas(categorical_df, numerical_df)
            
            print("")
            print(":: Pandas correlation : Categorical VS Categorical :: ")
            self.corr_method_pandas(categorical_df, categorical_df)
            
            final_df                     = self.merge_df(categorical_df, numerical_df, key)
            X                            = self.drop_columns(final_df, drop_columns)
            Y                            = final_df[target_column]
            return X, Y, final_df
        
        if data_type == 'Test':
            categorical_df, numerical_df = self._dataSplit_type(df, key)
            categorical_df               = self.onehot_encoding(categorical_df, key, data_type, custom_encode_col)
            
            print("")
            print(":: Pandas correlation : Categorical VS Numerical :: ")
            self.corr_method_pandas(categorical_df, numerical_df)
            
            print("")
            print(":: Pandas correlation : Categorical VS Categorical :: ")
            self.corr_method_pandas(categorical_df, categorical_df)
            
            final_df                     = self.merge_df(categorical_df, numerical_df, key)
            X                            = self.drop_columns(final_df, drop_columns)
            
            return X
            
        
                
                
        
        
            
            
        
    

