import hdbscan

def cluster_df(df_scale, df):
    '''
    This function performs the clustering on the dataframe df through the scaled data
    
    Parameters:
    -----------
    df_scale - the array produced from StandardScaler on the entire preprocessed dataframe df
    df - the preprocessed dataframe, prior to encoding
    
    Returns:
    ---------
    df with column 'cluster_id' to track cluster labels
    cls - the fit cluster object to make trees, etc.
    '''
    clusterer = hdbscan.HDBSCAN()
    cls = clusterer.fit(df_scale)
    df['cluster_id']=cls.labels_
    
    return cls, df