
# coding: utf-8

# In[1]:


def remove_nulls(data, percent_threshold):
    '''
    Removes cols 
    depending on the percentage
    threshold of missing data
    and imputes the the the rest
    with the mean of the col
    '''
    import pandas as pd
    import numpy as np

    df = pd.DataFrame.from_dict(data, orient='index')
   
    df = df.replace('NaN', np.nan)
    df_count = df.count()

    # remove na majority cols
    percent_missing = (1-df_count/146)
    percent_missing = percent_missing.sort_values()
    remained = percent_missing < percent_threshold
    df = df.loc[:, remained]
    # impute values 
    
    def fillWithMean(df):
        return df.fillna(df.mean()).dropna(axis=1, how='all')

    df =fillWithMean(df)
    
    # trans to dict
    final_dict = df.to_dict(orient='index')
    
    return final_dict

    

    

        

    



# In[2]:




