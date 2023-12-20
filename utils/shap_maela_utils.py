import numpy as np
import pandas as pd
import scipy.sparse as sparse
import argparse
from pathlib import Path
import shap
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from shap.plots import *
import json
import re

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#Defining function to give us DT model
def get_dt_model():
    model = DecisionTreeRegressor()
    return model

#Defining function to give us GBR model
def get_gbr_model():
    model = GradientBoostingRegressor()
    return model

#Defining function to give us RF model
def get_rf_model(no_jobs = 1):
    model = RandomForestRegressor(n_jobs = no_jobs, n_estimators = 10, min_samples_split = 4, 
                                  min_samples_leaf = 2, max_depth = 15)
    return model

#Defining function to give us XGB model
def get_xgb_model(no_jobs=1):
    model = xgb.XGBRegressor(objective = 'reg:squarederror', nthreads = no_jobs)
    return model


def train_model(x_train, y_train, model, no_epochs=100):
    model.fit(x_train, y_train)
        
def make_preds(model, x_test, y_test):
    preds = model.predict(x_test)
    return preds

# Read list to memory
def read_list(filename):
    # for reading also binary mode is important
    with open( filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list
    
def shap_main(X, y, reduced_snp_list):
        
    X = X.todense()
    
    #DT Model
    model = get_dt_model()
    train_model(X, y, model)
        
    #SHAP Bit
    explainer = shap.TreeExplainer(model, feature_names = reduced_snp_list, check_additivity=False)
    shap_test = explainer(X, check_additivity=False)
    
    shap_dt_df = pd.DataFrame(shap_test.values, columns=shap_test.feature_names)
    
#     shap_df.to_csv(output_folder+'shap_dt_dt.csv')
    
    #Creating dictionary for final csv with Normalized Values
    dt_dict = {}
    for col in shap_dt_df.columns:
        shap_dt_df[col] = shap_dt_df[col].apply(lambda x : abs(x))
        dt_dict[col] = shap_dt_df[col].mean()
    
#     shap.plots.bar(shap_test, show = False)
#     f = plt.gcf()
#     plt.title("SHAP DT_DT Values")
#     f.savefig("shap_dt_dt.png", bbox_inches = 'tight', dpi = 200)
    
    #GBR Model
    model = get_gbr_model()
    train_model(X, y, model)
    #SHAP Bit
    explainer = shap.TreeExplainer(model, feature_names = reduced_snp_list)
    shap_test = explainer(X)
    
    shap_gbr_df = pd.DataFrame(shap_test.values, columns=shap_test.feature_names)
    
#     shap_df.to_csv(output_folder+'shap_gbr_dt.csv')
    
    #Creating dictionary for final csv with Normalized Values
    gbr_dict = {}
    for col in shap_gbr_df.columns:
        shap_gbr_df[col] = shap_gbr_df[col].apply(lambda x : abs(x))
        gbr_dict[col] = shap_gbr_df[col].mean()
    
#     shap.plots.bar(shap_test, show = False)
#     f = plt.gcf()
#     plt.title("SHAP GBR_DT Values")
#     f.savefig("shap_gbr_dt.png", bbox_inches = 'tight', dpi = 200)
    
    
    #RF Model
    
    model = get_rf_model()
    train_model(X, y, model)
    #SHAP Bit
    explainer = shap.TreeExplainer(model, feature_names = reduced_snp_list)
    shap_test = explainer(X)
    
    shap_rf_df = pd.DataFrame(shap_test.values, columns=shap_test.feature_names)
    
#     shap_df.to_csv(output_folder+'shap_rf_dt.csv')
    
    #Creating dictionary for final csv with Normalized Values
    rf_dict = {}
    for col in shap_rf_df.columns:
        shap_rf_df[col] = shap_rf_df[col].apply(lambda x : abs(x))
        rf_dict[col] = shap_rf_df[col].mean()
    
#     shap.plots.bar(shap_test, show = False)
#     f = plt.gcf()
#     plt.title("SHAP RF_DT Values")
#     f.savefig("shap_rf_dt.png", bbox_inches = 'tight', dpi = 200)
    
    #XGB Model
    model = get_xgb_model()
    train_model(X, y, model)
    #SHAP Bit
    explainer = shap.TreeExplainer(model, feature_names = reduced_snp_list)
    shap_test = explainer(X)
    
    shap_xgb_df = pd.DataFrame(shap_test.values, columns=shap_test.feature_names)
    
    #Creating dictionary for final csv with Normalized Values
    xgb_dict = {}
    for col in shap_xgb_df.columns:
        shap_xgb_df[col] = shap_xgb_df[col].apply(lambda x : abs(x))
        xgb_dict[col] = shap_xgb_df[col].mean()
    
#     shap_df.to_csv(output_folder+'shap_xgb_dt.csv')
    
#     shap.plots.bar(shap_test, show = False)
#     f = plt.gcf()
#     plt.title("SHAP XGB_DT Values")
#     f.savefig("shap_xgb_dt.png", bbox_inches = 'tight', dpi = 200)
    
    #Creating a DF of all the Mean SHAP Dicts and Transposing to get Mean SHAPs as columns and Mutations as rows
    shap_df = pd.DataFrame([dt_dict, gbr_dict, rf_dict, xgb_dict]).transpose()
    shap_df.columns = ['DT', 'GBR', 'RF', 'XGB'] #Renaming columns
    #Normalizing Values to take Avg
    shap_df['Normalized_DT'] = (shap_df['DT'] - shap_df['DT'].mean())/shap_df['DT'].std()
    shap_df['Normalized_GBR'] = (shap_df['GBR'] - shap_df['GBR'].mean())/shap_df['GBR'].std()
    shap_df['Normalized_RF'] = (shap_df['RF'] - shap_df['RF'].mean())/shap_df['RF'].std()
    shap_df['Normalized_XGB'] = (shap_df['XGB'] - shap_df['XGB'].mean())/shap_df['XGB'].std()
    shap_df['Normalized_Average'] = shap_df[['Normalized_DT', 'Normalized_GBR', 'Normalized_RF', 'Normalized_XGB']].mean(axis=1)
    shap_df = shap_df.sort_values(by = 'Normalized_Average', ascending=False) #Sorting
    shap_df['Rank'] = range(1, len(shap_df)+1) #Rank Order for Spearman Correlation
    
    #Making Mutation_ID Column and reordering to bring it first
    shap_df['Mutation_ID'] = shap_df.index

    # Reorder the columns with 'Mutation_ID' as the first column
    shap_df = shap_df[['Mutation_ID'] + [col for col in shap_df if col != 'Mutation_ID']]
    
    return shap_df

#Here, we take i = 3 as we only care about the Log Normalized values of MIC
def data_prep_dt(data, csv, snp_list, drop_indels, i=3):
    labels = csv.iloc[:, i]
    index_list = list(csv.iloc[:, 4]) #4 is the column with the index list we need
    data_sliced = data[index_list, :]
    thresh = 0.0001
    
    #Drop indels from data_sliced
    if drop_indels:
        snp_array = np.array(snp_list)
        pattern = re.compile(r'.*indel.*')
        indel_columns = np.array([bool(pattern.match(col)) for col in snp_array])
        data_sliced = data_sliced[:, ~indel_columns]
        snp_list = snp_array[~indel_columns]
    
    model = DecisionTreeRegressor()
    selector = SelectFromModel(model, threshold = thresh)
    selector.fit(data_sliced, labels)
    reduced = selector.transform(data_sliced)
        
    reduced_snp_list = [b for a, b in zip(selector.get_support(), snp_list) if a]

    print(len(reduced_snp_list))

    return reduced.astype(np.float32), labels.astype(np.float32), reduced_snp_list
    
def dt_main(npz_data, mic, output_folder, snp_list, antibiotic, drop_indels):    
    X, y, reduced_snp_list = data_prep_dt(npz_data, mic, snp_list, drop_indels)
    shap_df = shap_main(X, y, reduced_snp_list)
    shap_df.to_csv(output_folder+antibiotic+'/'+'SHAP_DT_'+antibiotic+'.csv', index = False)
    return shap_df
    
def data_prep_rf(data, csv, snp_list, drop_indels, i=3):
    labels = csv.iloc[:, i]
    index_list = list(csv.iloc[:, 4]) #4 is the column with the index list we need
    data_sliced = data[index_list, :]
    thresh = 0.00006
    
    #Drop indels from data_sliced
    if drop_indels:
        snp_array = np.array(snp_list)
        pattern = re.compile(r'.*indel.*')
        indel_columns = np.array([bool(pattern.match(col)) for col in snp_array])
        data_sliced = data_sliced[:, ~indel_columns]
        snp_list = snp_array[~indel_columns]
    
    model = get_rf_model()
    selector = SelectFromModel(model, threshold = thresh)
    selector.fit(data_sliced, labels)
    reduced = selector.transform(data_sliced)
        
    reduced_snp_list = [b for a, b in zip(selector.get_support(), snp_list) if a]

    print(len(reduced_snp_list))

    return reduced.astype(np.float32), labels.astype(np.float32), reduced_snp_list
    
def rf_main(npz_data, mic, output_folder, snp_list, antibiotic, drop_indels):
    
    X, y, reduced_snp_list = data_prep_rf(npz_data, mic, snp_list, drop_indels)
    shap_df = shap_main(X, y, reduced_snp_list)
    shap_df.to_csv(output_folder+antibiotic+'/'+'SHAP_RF_'+antibiotic+'.csv', index = False)
    return shap_df

def data_prep_xgb(data, csv, snp_list, drop_indels, i=3):
    labels = csv.iloc[:, i]
    index_list = list(csv.iloc[:, 4]) #4 is the column with the index list we need
    data_sliced = data[index_list, :]
    thresh = 0.0175
    
    #Drop indels from data_sliced
    if drop_indels:
        snp_array = np.array(snp_list)
        pattern = re.compile(r'.*indel.*')
        indel_columns = np.array([bool(pattern.match(col)) for col in snp_array])
        data_sliced = data_sliced[:, ~indel_columns]
        snp_list = snp_array[~indel_columns]
    
    model = get_xgb_model()
    selector = SelectFromModel(model, threshold = thresh)
    selector.fit(data_sliced, labels)
    reduced = selector.transform(data_sliced)
        
    reduced_snp_list = [b for a, b in zip(selector.get_support(), snp_list) if a]

    print(len(reduced_snp_list))

    return reduced.astype(np.float32), labels.astype(np.float32), reduced_snp_list

def xgb_main(npz_data, mic, output_folder, snp_list, antibiotic, drop_indels):
    
    X, y, reduced_snp_list = data_prep_xgb(npz_data, mic, snp_list, drop_indels)
    shap_df = shap_main(X, y, reduced_snp_list)
    shap_df.to_csv(output_folder+antibiotic+'/'+'SHAP_XGB_'+antibiotic+'.csv', index = False)
    return shap_df

def gene_search(gene_sequence_file, mut_list):
    #First creating a gene_dict with positions of all known genes mapped to gene name
    with open(gene_sequence_file, mode = 'r') as f:
        gene_dict = {} #Creating a dictionary to store all genes
        for line in f:
            if line[0]=='>':

                vals = line.split()

                gene_flag = 0

                for val in vals:
                    if re.search('gene=', val): #Finding the gene name
                        gene_name = val.split('=')[1][:-1]
                        gene_flag = 1 #Some genes are unnamed (where gene_flag=0) so we just name them as gene3, gene4, etc.
                    if re.search('location=', val): #Finding the gene location
                        #Location/Position of Gene is in the syntax : 'location=<start>..<end>'
                        lower = int(re.sub("[^0-9]", "", val.split('.')[0]))
                        upper = int(re.sub("[^0-9]", "", val.split('.')[2]))

                if not gene_flag:
                    gene_name = line.split()[0]

                gene_dict[gene_name] = [lower, upper]

    dict_list = []
    for index,row in mut_list.iterrows():
        #coor_101 indicates position 101 in genome (1-indexed)
        coor = int(row['Mutation_ID'].split('_')[1])
        genes = [] 
        gene_pos = []
        #Loops through all the gene names to check. Can have multiple gene names per SNP because genes overlap
        #Stores gene name is genes and respective positions in gene_pos
        for key in gene_dict.keys():
            #If it falls in between the gene start and end index, then SNP is present in particular gene
            if coor>=gene_dict[key][0] and coor<=gene_dict[key][1]:
                genes.append(key)
                gene_pos.append(coor - gene_dict[key][0] + 1)
        if len(genes):
            dict_list.append({'SNP ID':'coor_'+str(coor), 'Gene' : re.sub('[\[\]\']', '', str(genes)), 'Gene_Position':re.sub('[\[\]\']', '', str(gene_pos))})
        else:
            #If SNP doesn't occur within a mapped gene, nan values are applied
            dict_list.append({'SNP ID':'coor_'+str(coor), 'Gene' : 'NA', 'Gene_Position': np.nan})
    output_df = pd.DataFrame.from_dict(dict_list)
    return output_df

def shap_result(df1, df2, df3, gene_sequence_file, drop_pe_ppe):
    
    df1['Mutation_ID'] = df1.index
    df2['Mutation_ID'] = df2.index
    df3['Mutation_ID'] = df3.index
    
    all_ids = sorted(set(df1['Mutation_ID']).union(df2['Mutation_ID'], df3['Mutation_ID']))

    # Sort DataFrames based on 'ID' column
    df1 = df1.sort_values(by='Mutation_ID')
    df2 = df2.sort_values(by='Mutation_ID')
    df3 = df3.sort_values(by='Mutation_ID')

    # Create a new dataframe with unique IDs
    result_df = pd.DataFrame({'Mutation_ID': all_ids})

    # Function to search for ID in a dataframe and return corresponding Col value
    def binary_search_and_get_value(df, target_id, col_name):
        left, right = 0, len(df) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_id = df['Mutation_ID'].iloc[mid]

            if mid_id == target_id:
                return df[col_name].iloc[mid]
            elif mid_id < target_id:
                left = mid + 1
            else:
                right = mid - 1
        return np.nan  # ID not found, return NA

    # Apply binary search to fill Col1, Col2, and Col3 in result_df
    result_df['DT_Shap'] = result_df['Mutation_ID'].apply(lambda x: binary_search_and_get_value(df1, x, 'Normalized_Average'))
    result_df['DT_Rank'] = result_df['Mutation_ID'].apply(lambda x: binary_search_and_get_value(df1, x, 'Rank'))
    result_df['RF_Shap'] = result_df['Mutation_ID'].apply(lambda x: binary_search_and_get_value(df2, x, 'Normalized_Average'))
    result_df['RF_Rank'] = result_df['Mutation_ID'].apply(lambda x: binary_search_and_get_value(df2, x, 'Rank'))
    result_df['XGB_Shap'] = result_df['Mutation_ID'].apply(lambda x: binary_search_and_get_value(df3, x, 'Normalized_Average'))
    result_df['XGB_Rank'] = result_df['Mutation_ID'].apply(lambda x: binary_search_and_get_value(df3, x, 'Rank'))
        
    result_df['Avg_Shap'] = result_df[['DT_Shap', 'RF_Shap', 'XGB_Shap']].sum(axis=1)/3
    result_df['Avg_Rank'] = result_df[['DT_Rank', 'RF_Rank', 'XGB_Rank']].sum(axis=1)/3
    
    # Replace NaN values with NA
    result_df.fillna("NA", inplace=True)
    
    mut_df = result_df[['Mutation_ID']].copy()
    gene_df = gene_search(gene_sequence_file, mut_df)
    result_df['Gene'] = gene_df['Gene']
    result_df['Gene_Position'] = gene_df['Gene_Position']
    
    if drop_pe_ppe:
        result_df = result_df[~result_df['Gene'].str.contains(r'(PE|PPE)')]
            
    # Reorder the columns with 'Mutation_ID' as the first column
    result_df = result_df[['Mutation_ID'] + [col for col in result_df if col != 'Mutation_ID']]
    
    return result_df.sort_values(by='Avg_Shap', ascending=False)
