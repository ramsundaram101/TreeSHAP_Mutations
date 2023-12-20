from utils.extract_utils import *
import argparse
import shutil

parser = argparse.ArgumentParser(description='Convert SNPs VCF to Binary Sparse Matrix')
parser.add_argument('--gz_file', default = './data/VARIANTS.csv.gz', type = str, help = 'path to SNP CSV.GZ file')
parser.add_argument('--output_folder', default='./extracted_data/', type=str, help='folder path to output NPZ file')
parser.add_argument('--output_json', default='mutations_cryptic.json', type=str, help='folder path to output Mutation JSON file')
args = parser.parse_args()

if __name__ == '__main__':
    gz_file = args.gz_file
    output_folder = args.output_folder
    output_json = output_folder + args.output_json 
    variant_file = unzip_gz(gz_file)
    shutil.copy(variant_file, output_folder + variant_file.split('/')[-1])
    sparse_matrix, shape, snp_list = convert_to_sparse_csv(variant_file)
    
    #Obtaining file name and saving the new compressed npz file in the output folder
    file_name = variant_file.split('/')[-1].split('.')[0]
    sparse.save_npz(output_folder+file_name+'_binary', sparse_matrix)
    write_list(snp_list, output_json)