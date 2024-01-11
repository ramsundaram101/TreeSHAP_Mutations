import numpy as np
from tqdm.auto import tqdm
import scipy.sparse as sparse
import gzip
import shutil
import re
import json
import pandas as pd

#Uncompressing the imported .gz file
#gz_file: path to the .gz file
def unzip_gz(gz_file):
    
    if gz_file.split('.')[-1] != 'gz':
        return gz_file
    
    with gzip.open( gz_file, 'rb') as f_in:
        file = re.sub('.gz$', '', gz_file)
        with open(file , 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            return file
        
#Converts the csv file to a sparse matrix (SNPs)
#variant_file: path to the variant file (.csv file)
def convert_to_sparse_csv(variant_file):
    
    #Defining two sets to store unique Mutations and Unique Mutation that occur more than once
    #We'll only consider mutations that occur multiple times
    unique_set = set()
    multi_unique_set = set()
    with open(variant_file, mode='r') as f:
        line = f.readline() #Skipping first column header line
        while line:
            line = f.readline()
            try: #Try-except because sometimes we get empty lines
                mut = line.split(',')[1]
            except:
                continue
            
            #Adds repeated mutation to multi_unique_set and new to unique_set
            if mut in unique_set:
                multi_unique_set.add(mut)
            else:
                unique_set.add(mut)
                
#     print(len(unique_set), len(multi_unique_set))
    
    #Creating a dictionary to store column indices for mutations to create sparse matrix
    mut_index_dict = {}
    counter = 0
    for mutation in multi_unique_set:
        mut_index_dict[mutation] = counter
        counter = counter + 1
        
    #Defining empty lists. These will store row & col index and data of non-zero entries respectively
    #This is how we create a csr sparse matrix
    rows = []
    cols = []
    data = []
    
    #Storing this to determine when moved on to next isolate i.e next row
    prev_iso = None
    row_no = -1
    
    
    #Iterating through lines in variant file
    with open(variant_file, mode='r') as var:
        line = var.readline() #Skipping first column header line
        while line:
            line = var.readline()
            
            #Avoiding empty lines
            if line.split(',')[0] == '':
                continue
            
            #Checking if it is the same isolate i.e same row no.
            if line.split(',')[0] != prev_iso:
                row_no = row_no + 1
                prev_iso = line.split(',')[0]
            
            try: #Try-except because sometimes we get empty lines
                curr_mutation = line.split(',')[1]
            except:
                continue
                
            try: #Will skip if mutation occurs only once
                cols.append(mut_index_dict[curr_mutation])
                rows.append(row_no)
                data.append(1)
            except:
                continue     
                    
    shape = (row_no+1, len(multi_unique_set))
    
    #Converting it to a scipy sparse matrix class and returning
    sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape = shape)
    return sparse_matrix, shape, list(mut_index_dict.keys())

#Converts the vcf file to a sparse matrix (SNPs)
#vcf_file: path to the vcf file (.vcf file)
def convert_to_sparse_vcf(vcf_file):
    
    #Defining empty lists. These will store row & col index and data of non-zero entries respectively
    #This is how we create a csr sparse matrix
    rows = []
    cols = []
    data = []
    
    #Adding SNP Labels
    snp_labels = []
    
    #Precomputing no. of lines in the vcf file for tqdm
    no_lines = line_count(vcf_file)-7
    
    #No. of skipped SNPs
    skipped = 0
    
    #Skipping the first 6 lines as they contain metadata, not info we need for the binary matrix
    vcf = open(vcf_file, mode='r')
    for i in range(6):
        line = vcf.readline()
        
    line = vcf.readline()
    no_rows = len(line.split())-9
    
    #Iterating through every line in the vcf file
    #Each line contains the SNP info and presence/absence of that SNP for the 3050 different sample genomes
    for i in range(no_lines):
        line = vcf.readline()
                
        #We save the positive examples for that particular SNP in pos_vals list
        pos_vals = []
        vals = line.split()
        
        for j in range(1, len(vals)):
            if vals[j]=="1/1":
                pos_vals.append(j-9) #We subtract 9 as first 9 strings are metadata abt the SNP
                
#         print(len(pos_vals))
        
        #Skipping ones that occur just once
        if len(pos_vals)<=1:
            skipped = skipped + 1
#             print("skipped")
            continue
    
        snp_labels.append(vals[2]) #Appending the SNP ID

        
        #We append positive values to the CSR Matrix lists
        for val in pos_vals:
            rows.append(val)
            cols.append(i - skipped)
            data.append(1)
                                   
    vcf.close()
        
    shape = (no_rows, no_lines - skipped)
    #Converting it to a scipy sparse matrix class and returning
    sparse_matrix = sparse.csr_matrix((data, (rows, cols)), shape = shape)
    return sparse_matrix, shape, snp_labels

#Function to write a list to a json file
#a_list: list to be written to json file
#output_json: path to desired output location of the json file
def write_list(a_list, output_json):
    with open(output_json, "w") as fp:
        json.dump(a_list, fp)

#Function to generate a dictionary of sample indexes and their corresponding sample IDs
#variant_file: path to the variant file (.csv file)
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

#Function to create MIC value dict of all samples
#mic_file: path to the MIC file (.csv file)
#variant_file: path to the variant file (.csv file)
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

#Function to count the number of lines in a file
#file: path to the file
def line_count(file):
    count = 0
    with open(file, mode='r') as f:
        line = f.readline()
        count+=1
        while line:
            line = f.readline()
            count+=1
    return count-1