from utils.extract_utils import *
import argparse

parser = argparse.ArgumentParser(description='Convert SNPs VCF to Binary Sparse Matrix')
parser.add_argument('--gz_file', default = './data/maela.vcf.gz', type = str, help = 'path to SNP VCF.GZ file')
parser.add_argument('--output_folder', default='./extracted_data/', type=str, help='folder path to output NPZ file')
parser.add_argument('--output_json', default='mutations_maela.json', type=str, help='folder path to output SNP JSON file')
args = parser.parse_args()

if __name__ == '__main__':
    #Parsing arguments
    gz_file = args.gz_file
    output_folder = args.output_folder
    output_json = output_folder + args.output_json
    
    #Unzipping and converting .vcf file to sparse matrix
    vcf_file = unzip_gz(gz_file)
    sparse_matrix, shape, snp_list = convert_to_sparse_vcf(vcf_file)
    
    #Obtaining file name and saving the new compressed npz file in the output folder
    file_name = vcf_file.split('/')[-1].split('.')[0]
    sparse.save_npz(output_folder+file_name+'_binary', sparse_matrix)
    write_list(snp_list, output_json)
