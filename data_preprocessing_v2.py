from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class Data_Preprocessing:
   
    def __init__(self):
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
    
    def label_encoding(self, categorical_df, key, data_type, custom_encode_col=None):
        '''
        categorical_df : Dataframe which contains only categorical columns
        '''
        df = categorical_df.copy()
        columns  = list(df.columns)
        columns.remove(key)
        print("** " + data_type + " **")
        for col in columns:
            '''if col == custom_encode_col :
                if data_type=='Train':
                    values = df[custom_encode_col].unique()
                    print(values)
                    self.values = values
                map_dict = {value: i + 1 for i, value in enumerate(self.values)}
                df[custom_encode_col] = df[custom_encode_col].map(map_dict)
                print(df)
            else:
            '''
            print("CATEGORICAL COLUMN ENCODED :: ", col)
            df[col] = self.label_encoder.fit_transform(df[col])
        
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
    
    def data_processing_pipeline(self, df, drop_columns , label_column, key, data_type = 'Train', custom_encode_col=None):
        '''
        df : Dataframe
        drop_columns : List of columns
        label_column : label column name or target column name
        data_type : Either Train or Test. 
                    Use Train : if DATASET is full i.e with label  
                    Use Test:  if DATASET wihout label.
        key : key is column name through which we merge dataframe
        custom_encode_col : column name use for to encode with custom define encode values.
        
        RETURN 1. TRAIN ---> X,Y
               2. TEST  ---> X
        '''
        if data_type == 'Train':
            categorical_df, numerical_df = self._dataSplit_type(df, key)
            categorical_df               = self.label_encoding(categorical_df, key, data_type, custom_encode_col)
            final_df                     = self.merge_df(categorical_df, numerical_df, key)
            X = self.drop_columns(final_df, drop_columns)
            Y = final_df[label_column]
            return X, Y
        
        if data_type == 'Test':
            categorical_df, numerical_df = self._dataSplit_type(df, key)
            categorical_df               = self.label_encoding(categorical_df, key, data_type, custom_encode_col)
            final_df                     = self.merge_df(categorical_df, numerical_df, key)
            X = self.drop_columns(final_df, drop_columns)
            
            return X
            
        
                
                
        
        
            
            
        
    
