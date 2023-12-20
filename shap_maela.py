from utils.shap_maela_utils import *
import argparse

parser = argparse.ArgumentParser(description='Linear Reg on Dataset')
parser.add_argument('--npz_file', default='./maela_binary.npz', type = str, help = 'path to SNPs NPZ file')
parser.add_argument('--output_folder', default='./Outputs/Shap/', type=str, help='folder path to output predictions')
parser.add_argument('--json_file',default = './mutations_maela.json', type = str, help = 'path to SNPs name JSON file')
parser.add_argument('--gene_sequence_file', default = './sequence.txt', type = str, help = 'folder path to the gene sequence FASTA format txt file')
parser.add_argument('--raw_mic_file', default='./pen_mics_maela_snps.csv', type=str, help='folder path to raw MIC File')
parser.add_argument('--antibiotic', default='PCN', type=str, help='Antibiotic to Extract')
parser.add_argument('--drop_indels', default=False, type=str2bool, help='Choice to Drop Indels in pre-processing')
parser.add_argument('--drop_pe_ppe', default=False, type=str2bool, help='Choice to Drop Mutations in PE/PPE Genes')
args = parser.parse_args()

if __name__ == '__main__':
    npz_file = args.npz_file
    output_folder = args.output_folder
    json_file = args.json_file
    gene_sequence_file = args.gene_sequence_file
    mic_file = args.raw_mic_file
    antibiotic = args.antibiotic
    drop_indels = args.drop_indels
    drop_pe_ppe = args.drop_pe_ppe
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(output_folder+antibiotic).mkdir(parents=True, exist_ok=True)
    
    mic_df = pd.read_csv(mic_file)
    npz_data = sparse.load_npz(npz_file)
    snp_list = read_list(json_file)
    
    shap_dt_df = dt_main(npz_data, mic_df, output_folder, snp_list, antibiotic, drop_indels)
    shap_rf_df = rf_main(npz_data, mic_df, output_folder, snp_list, antibiotic, drop_indels)
    shap_xgb_df = xgb_main(npz_data, mic_df, output_folder, snp_list, antibiotic, drop_indels)
        
    output_df = shap_result(shap_dt_df, shap_rf_df, shap_xgb_df, gene_sequence_file, drop_pe_ppe)
    output_df.to_csv(output_folder+antibiotic+'/'+'SHAP_Avg_'+antibiotic+'.csv', index = False)