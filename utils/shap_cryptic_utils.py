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

#Defining function to parse arguments, converting to bool if necessary
#v: argument to be converted to bool
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
#no_jobs: number of jobs to run in parallel
def get_rf_model(no_jobs = 1):
    model = RandomForestRegressor(n_jobs = no_jobs, n_estimators = 10, min_samples_split = 4, 
                                  min_samples_leaf = 2, max_depth = 15)
    return model

#Defining function to give us XGB model
#no_jobs: number of jobs to run in parallel
def get_xgb_model(no_jobs=1):
    model = xgb.XGBRegressor(objective = 'reg:squarederror', nthreads = no_jobs)
    return model

#Function to train the desired ML model
#x_train: training data
#y_train: training labels
#model: ML model to be trained
#no_epochs: number of epochs to train for
def train_model(x_train, y_train, model, no_epochs=100):
    model.fit(x_train, y_train)
        
#Function to make predictions on test data
#model: ML model to be used for prediction
#x_test: test data
#y_test: test labels
def make_preds(model, x_test, y_test):
    preds = model.predict(x_test)
    return preds

# Read list to memory from json file
# filename: path to json file
def read_list(filename):
    # for reading also binary mode is important
    with open( filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list

#Main SHAP Loop to get SHAP Values across all 4 Tree-Based ML Models
#X: Training Data (reduced sparse matrix)
#y: Training Labels (df)
#reduced_snp_list: List of SNPs selected by feature selection for which we want SHAP values
#output_folder: folder path to output predictions
#data_model: data model to be used for feature selection
#raw_files: bool value to toggle whether to output raw Shapiro values (toggle to True in args given to the .py file to output raw SHAP values)
#Uncomment the 4 lines below the shap plots function to add plots to the output folder
def shap_main(X, y, reduced_snp_list, output_folder, data_model, raw_files):
        
    X = X.todense()
    
    #DT Model
    model = get_dt_model()
    train_model(X, y, model)
        
    #SHAP Bit
    explainer = shap.TreeExplainer(model, feature_names = reduced_snp_list, check_additivity=False)
    shap_test = explainer(X, check_additivity=False)
    
    shap_dt_df = pd.DataFrame(shap_test.values, columns=shap_test.feature_names)
    
    if raw_files:
        shap_df.to_csv(output_folder+'shap_dt_'+data_model+'.csv')
    
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

    if raw_files:    
        shap_df.to_csv(output_folder+'shap_gbr_'+data_model+'.csv')
        
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
    
    if raw_files:
        shap_df.to_csv(output_folder+'shap_rf_'+data_model+'.csv')

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
    if raw_files:
        shap_df.to_csv(output_folder+'shap_xgb_'+data_model+'.csv')
    
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

#Function to prepare data for SHAP by selecting samples for which we have MIC values, performing feature selection via DT and returning the reduced data
#data: data to be used for feature selection (npz data)
#csv: csv file containing MIC values
#snp_list: list of SNP (can include indels) names
#drop_indels: bool value to drop indels from the data (toggle to True in args given to the .py file to drop indels)
#i: column number of MIC values in the csv file (7 for cryptic)
#Here, we take i = 7 as we only care about the Log Normalized values of MIC
def data_prep_dt(data, csv, snp_list, drop_indels, i=7):
    labels = csv.iloc[:, i]
    index_list = list(csv.iloc[:, 1]) #1 is the column with the index list we need
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

    return reduced.astype(np.float32), labels.astype(np.float32), reduced_snp_list

#Main loop for DT selected data
#npz_data: data to be used for feature selection (sparse matrix)
#mic: df containing MIC values
#output_folder: folder path to output predictions
#snp_list: list of SNP (can include indels) names
#antibiotic: antibiotic of interest
#drop_indels: bool value to drop indels from the data (toggle to True in args given to the .py file to drop indels)
#raw_files: bool value to output raw SHAP values (toggle to True in args given to the .py file to output raw SHAP values)
def dt_main(npz_data, mic, output_folder, snp_list, antibiotic, drop_indels, raw_files):    
    X, y, reduced_snp_list = data_prep_dt(npz_data, mic, snp_list, drop_indels)
    shap_df = shap_main(X, y, reduced_snp_list, output_folder, 'dt', raw_files)
    shap_df.to_csv(output_folder+antibiotic+'/'+'SHAP_DT_'+antibiotic+'.csv', index = False)
    return shap_df

#Function to prepare data for SHAP by selecting samples for which we have MIC values, performing feature selection via RF and returning the reduced data
#data: data to be used for feature selection (sparse matrix)
#csv: df containing MIC values
#snp_list: list of SNP (can include indels) names
#drop_indels: bool value to drop indels from the data (toggle to True in args given to the .py file to drop indels)
#i: column number of MIC values in the csv file (7 for cryptic)
#Here, we take i = 7 as we only care about the Log Normalized values of MIC
def data_prep_rf(data, csv, snp_list, drop_indels, i=7):
    labels = csv.iloc[:, i]
    index_list = list(csv.iloc[:, 1]) #1 is the column with the index list we need
    data_sliced = data[index_list, :]
    thresh = 0.000005
    
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

    return reduced.astype(np.float32), labels.astype(np.float32), reduced_snp_list

#Main loop for RF selected data
#npz_data: data to be used for feature selection (sparse matrix)
#mic: df containing MIC values
#output_folder: folder path to output predictions
#snp_list: list of SNP (can include indels) names
#antibiotic: antibiotic of interest
#drop_indels: bool value to drop indels from the data (toggle to True in args given to the .py file to drop indels)
#raw_files: bool value to output raw SHAP values (toggle to True in args given to the .py file to output raw SHAP values)
def rf_main(npz_data, mic, output_folder, snp_list, antibiotic, drop_indels, raw_files):
    
    X, y, reduced_snp_list = data_prep_rf(npz_data, mic, snp_list, drop_indels)
    shap_df = shap_main(X, y, reduced_snp_list, output_folder, 'rf', raw_files)
    shap_df.to_csv(output_folder+antibiotic+'/'+'SHAP_RF_'+antibiotic+'.csv', index = False)
    return shap_df

#Function to prepare data for SHAP by selecting samples for which we have MIC values, performing feature selection via XGB and returning the reduced data
#data: data to be used for feature selection (sparse matrix)
#csv: df containing MIC values
#snp_list: list of SNP (can include indels) names
#drop_indels: bool value to drop indels from the data (toggle to True in args given to the .py file to drop indels)
#i: column number of MIC values in the csv file (7 for cryptic)
#Here, we take i = 7 as we only care about the Log Normalized values of MIC
def data_prep_xgb(data, csv, snp_list, drop_indels, i=7):
    labels = csv.iloc[:, i]
    index_list = list(csv.iloc[:, 1]) #1 is the column with the index list we need
    data_sliced = data[index_list, :]
    thresh = 0.00005
    
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

    return reduced.astype(np.float32), labels.astype(np.float32), reduced_snp_list

#Main loop for XGB selected data
#npz_data: data to be used for feature selection (sparse matrix)
#mic: df containing MIC values
#output_folder: folder path to output predictions
#snp_list: list of SNP (can include indels) names
#antibiotic: antibiotic of interest
#drop_indels: bool value to drop indels from the data (toggle to True in args given to the .py file to drop indels)
#raw_files: bool value to output raw SHAP values (toggle to True in args given to the .py file to output raw SHAP values)
def xgb_main(npz_data, mic, output_folder, snp_list, antibiotic, drop_indels, raw_files):
    
    X, y, reduced_snp_list = data_prep_xgb(npz_data, mic, snp_list, drop_indels)
    shap_df = shap_main(X, y, reduced_snp_list, output_folder, 'xgb', raw_files)
    shap_df.to_csv(output_folder+antibiotic+'/'+'SHAP_XGB_'+antibiotic+'.csv', index = False)
    return shap_df

#Function to create sample dictionary including all unique samples and their indices
#variant_file: path to variant file (.csv file)
def sample_dict(variant_file):
    index_dict = {} #Dict to store sample indexes
    var_gene_dict = {} #Dict to store sample genes and variants
    var_df = pd.read_csv(variant_file)
    prev_id = ""
    counter = 0
    
    #Iterating through rows and storing all unique samples with indices from 0 onwards
    for index, row in var_df.iterrows():
        if row["UNIQUEID"] == prev_id: #If sample already stored in dict, continue
            continue
        if pd.isna(row['GENE']):
            gene = 'NA'
        else:
            gene = row['GENE']
        index_dict[counter] = row["UNIQUEID"]
        var_gene_dict[counter] = [row['VARIANT'], gene]
        prev_id = row["UNIQUEID"]
        counter = counter + 1
        
    return index_dict, var_gene_dict

#Function to create MIC value dict
#mic_file: path to MIC file (.csv file)
#variant_file: path to variant file (.csv file)
#antibiotic: antibiotic of interest to be mapped
def read_mic(mic_file, variant_file, antibiotic):
    sd, vgd = sample_dict(variant_file) #Index Dictionary of all Samples
    md = {}
        
    #Storing index list with those indexes that have a MIC value
    index_list = []
    
    #Empty DF to store MIC Values
    df = pd.DataFrame(columns = [antibiotic+'_MIC','Index', 'Sample', ])
    mic_df = pd.read_csv(mic_file)
    
    #Iterating through MIC File and Storing all Samples and their MIC values
    for index, row in mic_df.iterrows():
        if pd.isna(row[antibiotic+"_MIC"]):
            continue
        md[row["UNIQUEID"]] = float(re.sub("[<>=]", "", row[antibiotic+"_MIC"]))
    
    #Matching those samples with MIC Values to those samples in the original dataset
    for ind in sd.keys():
        if(sd[ind] in md):
            df = df.append({antibiotic+"_MIC" : md[sd[ind]], 'Index' : ind, 'Sample' : sd[ind], 'Variant' : vgd[ind][0],
                           'Gene' : vgd[ind][1]}, ignore_index=True)
            index_list.append(ind)
    
    #Creating Log Normalized MIC Values
    df["MIC_Normalized"] = (df[antibiotic+"_MIC"] - df[antibiotic+"_MIC"].mean()) / df[antibiotic+"_MIC"].std()
    df["Log_MIC"] = np.log2(df[antibiotic+"_MIC"].astype('float64'))
    df["Log_MIC_Normalized"] = (df["Log_MIC"] - df["Log_MIC"].mean()) / df["Log_MIC"].std()
    return df, index_list

#Function to look up gene for each mutation
#var_df: variant dataframe
#mut_list: list of mutations
def gene_search(var_df, mut_list):
    dict_list = []
    var_df = var_df.sort_values(by=['VARIANT'])
    for index,row in mut_list.iterrows():
        var_ind = var_df['VARIANT'].searchsorted(row['Mutation_ID'])
        var_gene = var_df.iloc[var_ind]['GENE']
        if var_gene is np.nan:
            var_gene = 'NA'

        new_row = {'Mutation_ID' : row['Mutation_ID'], 'Gene' : var_gene}
        dict_list.append(new_row)
        
    output_df = pd.DataFrame.from_dict(dict_list)
    return output_df

#Function to create final output dataframes of shap values for each mutation
#df1: shap output df from DT model
#df2: shap output df from RF model
#df3: shap output df from XGB model
#var_df: variant dataframe
#drop_pe_ppe: bool value to drop mutations in PE/PPE genes (toggle to True in args given to the .py file to drop mutations in PE/PPE genes)
def shap_result(df1, df2, df3, var_df, drop_pe_ppe):
    
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
    gene_df = gene_search(var_df, mut_df)
    result_df['Gene'] = gene_df['Gene']
    
    if drop_pe_ppe:
        result_df = result_df[~result_df['Gene'].str.contains(r'(PE|PPE)')]
            
    # Reorder the columns with 'Mutation_ID' as the first column
    result_df = result_df[['Mutation_ID'] + [col for col in result_df if col != 'Mutation_ID']]
    
    return result_df.sort_values(by='Avg_Shap', ascending=False)
