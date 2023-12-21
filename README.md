# TreeSHAP_Mutations
This repository contains code for a pipeline that ultimately utilises Tree-Based Interpretable ML and TreeSHAP to identify and interpret antibiotic resistance mechanisms in pathogenic bacteria. 
We return a `.csv` file that contains the mutations in order of their SHAPLey values (signifying their importance) towards antibiotic resistance. 

This repository includes data from two different datasets:
1. The `maela` dataset, containing SNP data of *Streptococcus pneumoniae*
2. The `cryptic` dataset, containing SNP and indel data of *Mycobacterium tuberculosis*

Here's a brief description of the repository:
1. `extract_maela.py` and `extract_cryptic.py` are used to extract the required mutation data from the original Whole Genome Sequencing data and convert it into a format our pipeline can use, namely a sparse matrix format, i.e. a `.npz` file
2. `shap_maela.py` and `shap_cryptic.py` run the pipeline on their respective datasets
3. The `data` folder contains the original WGS data
4. The `extracted_data` folder contains pre-extracted `.npz` files. However, you can run the extract .py files nonetheless.
5. The `utils` folder contains Python files that are called by the rest of the code

Here's how to use the files in this repository after cloning it.

## Extraction

**Note:** You can skip this step if you'd like, as we've already stored the extracted versions in `extracted_data`

### Extracting maela

`python extract_maela.py --gz_file ./data/maela.vcf.gz --output_json mutations_maela.json`

### Extracting cryptic

`python extract_cryptic.py --gz_file ./data/VARIANTS_SAMPLE.csv.gz --output_json mutations_cryptic.json`

**Note:** If you're using the original larger VARIANTS file, remove the `_Sample`

## ML/Shap Bit

### Shap maela

`python shap_maela.py --npz_file ./extracted_data/maela_binary.npz --json_file ./extracted_data/mutations_maela.json --gene_sequence_file ./data/sequence.txt --raw_mic_file ./data/pen_mics_maela_snps.csv`

### Shap cryptic

`python shap_cryptic.py --npz_file ./extracted_data/VARIANTS_SAMPLE_binary.npz --json_file ./extracted_data/mutations_cryptic.json --variant_file ./data/VARIANTS_SAMPLE.csv --raw_mic_file ./data/CRyPTIC_reuse_table_20221019.csv --antibiotic MXF --drop_indels False --drop_pe_ppe True`

**Note:**
1. You can run the algorithm on any of the 13 antibiotics in `CRyPTIC_reuse_table_20221019.csv` by changing the `antibiotic` argument
2. You can remove indels by toggling `drop_indels` to `True`
3. You can drop all PE and PPE genes by toggling `drop_pe_ppe` to `True`
