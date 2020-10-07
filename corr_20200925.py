# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:00:06 2020

@author: whyang
"""

# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

#####################
# declare functions #
#####################
##
# remove leading and trailing characters of each value across all cells in dataframe
def trim_all_cells(df):
    # trim whitespace from ends of each value across all series in dataframe
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return df.applymap(trim_strings)

def heatmap(x, y, size, corr):
    ###
    # heatmap 1: demonstrate the correlation of each two features in terms of the size of correlated ratio (position/negative)
    ##
    fig, ax = plt.subplots(figsize=(16, 14))
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]

    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}
    
    #sns.set(font=['sans-serif'])
    size_scale = 300
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right', fontsize=16)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_title('圖書書目清單 (Feature Correlation)')
    ax.set_xlabel('特徵')
    ax.set_ylabel('特徵')
    plt.show() # display the graph
    
    ###
    # heatmap 2: demonstrate the correlation of each two features in terms of the correlated ratio
    ##
    fig, ax1 = plt.subplots(figsize=(16,8))
    corr = corr.pivot('x', 'y', 'value')
    ax1 = sns.heatmap(corr, vmax=1, vmin=-1, cmap='coolwarm', center=0, robust=True,
                     annot=True, annot_kws={'size':14}, fmt='.1f',
                     linewidths=0.5, square=True)
    ax1.set_xticklabels(ax1.get_yticklabels(), rotation=45, fontsize=16)
    ax1.set_title('圖書書目清單 (Feature Correlation)')
    ax1.set_xlabel('特徵')
    ax1.set_ylabel('特徵')
    plt.show()

def preprocess(base_dir):
    ###
    # step 1: read into the booklist's content
    ##
    booklist = os.path.join(base_dir, 'booklist.xlsx') # the configuration file
    df = pd.read_excel(booklist,
                       usecols=['書目系統號', '書刊名', '出版項', '出版年', '簡繁體代碼', '標題', '出版社',
                                '作者', 'ISBN', '領域別', '摘要', '索書號', '分類號'])                       
    trim_all_cells(df) # remove leading and tailing white space of string (content of cell in dataframe)
        
    #統計為空的數目
    print(df.isnull().sum(axis = 0))
    
    ###
    # step 2: replacing all NULL values in the dataframe of booklist with na 
    ##
    df.fillna('na', inplace = True)  # df['書目系統號'].fillna('na', inplace = True)
    
    ###
    # step 3: 利用LabelEncoder編碼每個attribute
    ##
    class_le = LabelEncoder() # construct LabelEncoder
    
    # '書目系統號'
    print('書目系統號')
    for idx, label in enumerate(df['書目系統號']):
        pass #print(idx, label)
    #df['書目系統號'] = class_le.fit_transform((df['書目系統號'].values).astype(str))
    df['書目系統號'] = class_le.fit_transform(df['書目系統號'].astype(str))
    
    # '書刊名'
    print('書刊名')
    for idx, label in enumerate(df['書刊名']):
        pass #print(idx, label)
    df['書刊名'] = class_le.fit_transform(df['書刊名'].astype(str))
    
    # '出版項' 
    print('出版項')
    for idx, label in enumerate(df['出版項']):
        pass #print(idx, label)
    df['出版項'] = class_le.fit_transform(df['出版項'].astype(str))  
    
    # '簡繁體代碼'
    print('簡繁體代碼')
    for idx, label in enumerate(df['簡繁體代碼']):
        pass #print(idx, label)
    df['簡繁體代碼'] = class_le.fit_transform(df['簡繁體代碼'].astype(str))
    
    # '標題'
    print('標題')
    for idx, label in enumerate(df['標題']):
        pass #print(idx, label)
    df['標題'] = class_le.fit_transform(df['標題'].astype(str))    
    
    # '出版社'
    print('出版社')
    for idx, label in enumerate(df['出版社']):
        pass #print(idx, label)
    df['出版社'] = class_le.fit_transform(df['出版社'].astype(str))
    
    # '作者'
    print('作者')
    for idx, label in enumerate(df['作者']):
        pass #print(idx, label)
    df['作者'] = class_le.fit_transform(df['作者'].astype(str))  
    
    # 'ISBN'
    print('ISBN')
    for idx, label in enumerate(df['ISBN']):
        pass #print(idx, label)
    df['ISBN'] = class_le.fit_transform(df['ISBN'].astype(str)) 

    # '領域別'
    print('領域別')
    for idx, label in enumerate(df['領域別']):
        pass #print(idx, label)
    df['領域別'] = class_le.fit_transform(df['領域別'].astype(str))     

    # '摘要'
    print('摘要')
    for idx, label in enumerate(df['摘要']):
        pass #print(idx, label)
    df['摘要'] = class_le.fit_transform(df['摘要'].astype(str))     

    # '索書號'
    print('索書號')
    for idx, label in enumerate(df['索書號']):
        pass #print(idx, label)
    df['索書號'] = class_le.fit_transform(df['索書號'].astype(str))   
    
    # '分類號'
    print('分類號')
    for idx, label in enumerate(df['分類號']):
        pass #print(idx, label)
    df['分類號'] = class_le.fit_transform(df['分類號'].astype(str))   
            
    #統計為空的數目
    print('after drop nan', df.isnull().sum(axis = 0))
    df.dropna(inplace=True) # omit the row of data in which any NAN value is contained
    
    # keep booklist after data cleansing
    booklist = os.path.join(base_dir, 'booklist_cleansing.xlsx') # the configuration file
    df.to_excel(booklist, sheet_name='cleansed', index=False)
    
################
# main program #
################
if __name__ == '__main__':
    # Register converters to avoid warnings
    pd.plotting.register_matplotlib_converters()
    plt.rc("figure", figsize=(16,14))
    plt.rc("font", size=16)
    plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題(正黑體)    

    # anchor the folder's path of the booklist file
    base_dir = os.path.dirname(__file__)  
    
    ###
    # conduct preprocessing to the data of booklist if needs
    # input: booklist.xlsx
    # output: booklist_cleansing.xlsx
    ##
    preprocess(base_dir)
    
    ###
    # read into the booklist's contents which are fulfilling data cleansing process
    ##
    booklist = os.path.join(base_dir, 'booklist_cleansing.xlsx') # the configuration file
    df = pd.read_excel(booklist,
                       usecols=['書目系統號', '書刊名', '出版項', '出版年', '簡繁體代碼', '標題', '出版社',
                                '作者', 'ISBN', '領域別', '摘要', '索書號', '分類號'])                        
    trim_all_cells(df) # remove leading and tailing white space of string (content of cell in dataframe)
        
    # 統計為空的數目
    print(df.isnull().sum(axis = 0))    

    # normalizing each attribute
    sc = MinMaxScaler(feature_range=(-1, 1))
    df['書目系統號'] = sc.fit_transform(df[['書目系統號']])
    df['書刊名'] = sc.fit_transform(df[['書刊名']])
    df['出版項'] = sc.fit_transform(df[['出版項']])
    df['出版年'] = sc.fit_transform(df[['出版年']])
    df['簡繁體代碼'] = sc.fit_transform(df[['簡繁體代碼']])
    df['標題'] = sc.fit_transform(df[['標題']])
    df['出版社'] = sc.fit_transform(df[['出版社']])
    df['作者'] = sc.fit_transform(df[['作者']])
    df['ISBN'] = sc.fit_transform(df[['ISBN']])
    df['領域別'] = sc.fit_transform(df[['領域別']])
    df['摘要'] = sc.fit_transform(df[['索書號']])
    df['索書號'] = sc.fit_transform(df[['索書號']])
    df['分類號'] = sc.fit_transform(df[['分類號']])
    
    ###
    # step 1: observe the correlation between any two features
    ##       
    columns =  ['書目系統號', '書刊名', '出版項', '出版年', '簡繁體代碼', '標題', '出版社', '作者', 'ISBN',
                '領域別', '摘要', '索書號', '分類號']
    corr = df[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']
    heatmap(x=corr['x'], y=corr['y'], size=corr['value'].abs(), corr=corr)               

    # clustering
    silhouette_avgs = []
    _select = 2
    _score = 0
    _labels = []
    _cluster_centers = []
    _maxCluster = 21 # the maximal number of clusters
    ks = range(2, _maxCluster) 
    
    # loop of evaluating process
    for k in ks:
        kmeans = KMeans(n_clusters = k)
        #kmeans.fit(df[['書目系統號', '書刊名', '出版項', '標題', '出版社', '作者', 'ISBN', '索書號', '分類號']])
        kmeans.fit(df[['書目系統號', '書刊名', '出版項', '出版社', '作者', '索書號']])
        cluster_labels = kmeans.labels_ # get cluster's labels after conducting KMeans
        #silhouette_avg = metrics.silhouette_score(df[['書目系統號', '書刊名', '出版項', '標題', '出版社', '作者', 'ISBN', '索書號', '分類號']],
        silhouette_avg = metrics.silhouette_score(df[['書目系統號', '書刊名', '出版項', '出版社', '作者', '索書號']],
                                                  cluster_labels)
        cluster_centers = kmeans.cluster_centers_ # get cluster's center after conducting KMeans
        silhouette_avgs.append(silhouette_avg) # record evaluated silhouette value       
        if silhouette_avg > _score:
            _select = k
            _score = silhouette_avg
            _labels = cluster_labels
            _cluster_centers = cluster_centers          

    # 做圖並印出 k = 2 到 _maxCluster 的績效
    plt.bar(ks, silhouette_avgs)
    df['cluster'] = _labels
    plt.show()
    print(silhouette_avgs)
    print('---------------------')
 
    #由績效圖找到在k群的績效比較好，選擇 k groups        
    print('_select = ', _select)
    print('_score = ', _score)
    print('_labels = ', _labels)
    print('_cluster_centers = ', _cluster_centers)      

    df['label'] = _labels

    ###    
    # keep booklist after clustering
    ##
    booklist = os.path.join(base_dir, 'booklist_cluster.xlsx') # the configuration file
    df.to_excel(booklist, sheet_name='cluster', index=False)
    
    ###
    # display figures for observing clustering distribution
    ##
    #plt.figure(figsize=(16, 8))
    figure_dir = os.path.join(base_dir, 'figure') 

    pd.plotting.register_matplotlib_converters()
    plt.rc("figure", figsize=(16,14))
    plt.rc("font", size=20)
    plt.rcParams['axes.unicode_minus'] = False # 修復負號顯示問題(正黑體)   
    
    # 作者_索書號
    plt.xlabel('作者')
    plt.ylabel('索書號')
    plt.scatter(df['作者'], df['索書號'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '作者_索書號.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 出版社_索書號
    plt.xlabel('出版社')
    plt.ylabel('索書號')
    plt.scatter(df['出版社'], df['索書號'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '出版社_索書號.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 出版社_作者
    plt.xlabel('出版社')
    plt.ylabel('作者')
    plt.scatter(df['出版社'], df['作者'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '出版社_作者.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 出版項_索書號
    plt.xlabel('出版項')
    plt.ylabel('索書號')
    plt.scatter(df['出版項'], df['索書號'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '出版項_索書號.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 出版項_作者
    plt.xlabel('出版項')
    plt.ylabel('作者')
    plt.scatter(df['出版項'], df['作者'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '出版項_作者.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 出版項_出版社
    plt.xlabel('出版項')
    plt.ylabel('出版社')
    plt.scatter(df['出版項'], df['出版社'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '出版項_出版社.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書刊名_索書號
    plt.xlabel('書刊名')
    plt.ylabel('索書號')
    plt.scatter(df['書刊名'], df['索書號'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書刊名_索書號.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書刊名_作者
    plt.xlabel('書刊名')
    plt.ylabel('作者')
    plt.scatter(df['書刊名'], df['作者'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書刊名_作者.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書刊名_出版社
    plt.xlabel('書刊名')
    plt.ylabel('出版社')
    plt.scatter(df['書刊名'], df['出版社'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書刊名_出版社.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書刊名_出版項
    plt.xlabel('書刊名')
    plt.ylabel('出版項')
    plt.scatter(df['書刊名'], df['出版項'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書刊名_出版項.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書目系統號_索書號
    plt.xlabel('書目系統號')
    plt.ylabel('索書號')
    plt.scatter(df['書目系統號'], df['索書號'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書目系統號_索書號.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書目系統號_作者
    plt.xlabel('書目系統號')
    plt.ylabel('作者')
    plt.scatter(df['書目系統號'], df['作者'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書目系統號_作者.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
    # 書目系統號_出版社
    plt.xlabel('書目系統號')
    plt.ylabel('出版社')
    plt.scatter(df['書目系統號'], df['出版社'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書目系統號_出版社.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()


    # 書目系統號_出版項
    plt.xlabel('書目系統號')
    plt.ylabel('出版項')
    plt.scatter(df['書目系統號'], df['出版項'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書目系統號_出版項.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()

    # 書目系統號_書刊名
    plt.xlabel('書目系統號')
    plt.ylabel('書刊名')
    plt.scatter(df['書目系統號'], df['書刊名'], c=df['label']) #C是第三維度 以顏色做維度
    _figure = os.path.join(figure_dir, '書目系統號_書刊名.jpg')
    plt.savefig(_figure, dpi=300)
    plt.show()
    
###
# end of file
##