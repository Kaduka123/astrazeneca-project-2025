# CHO Analysis - Technical Requirements

## Project Overview

This document defines the technical requirements for the CHO (Chinese Hamster Ovary) cell line RNA-seq data analysis project. The goal is to analyze expression data from CHO cell lines to identify gene markers for clone selection and characterize sequence features of consistently expressed genes.

## Data Sources

The project uses the following input data files:

1. **expression.txt**: Tab-separated file containing expression values of genes in transcript per million (TPM). Genes are in rows and samples in columns.
2. **MANIFEST.txt**: Tab-separated file containing metadata for each sample derived from SRA.
3. **CDS_sequences.fasta**: FASTA file containing coding sequence parts of protein-coding genes derived from Ensembl.
4. **5UTR_sequences.fasta**: FASTA file containing 5' UTR sequences of protein-coding genes derived from Ensembl.
5. **3UTR_sequences.fasta**: FASTA file containing 3' UTR sequences of protein-coding genes derived from Ensembl.

## Core Requirements

### Data Preprocessing Requirements

1. **Data Loading**
   - Implement efficient loading of all data files (expression data, sequence files, and metadata)
   - Validate data integrity upon loading

2. **Batch Effect Correction**
   - Implement batch effect correction for expression data
   - Account for different experimental conditions across laboratories and cell lines
   - Use appropriate statistical methods (e.g., ComBat, limma, or other established methods)
   - Validate effectiveness of batch correction

3. **Data Normalization**
   - Ensure expression data is properly normalized (data is already in TPM, but additional normalization may be required)
   - Handle missing values appropriately

## Task 1: Identify Gene Markers for Clone Selection

### Task 1 Requirements

1. **Correlation Analysis**
   - Calculate correlation between each gene's expression and the final product (LC) expression
   - Implement multiple correlation methods (e.g., Pearson, Spearman)
   - Account for potential non-linear relationships

2. **Marker Gene Ranking**
   - Develop a robust ranking system for potential marker genes
   - Consider multiple factors in ranking (correlation strength, expression stability, biological relevance)
   - Prioritize genes with strong and consistent correlation with LC across samples

3. **Visualization**
   - Create visualizations showing the relationship between marker gene expression and LC expression
   - Generate plots for top-ranked marker genes
   - Create summary visualizations of marker gene characteristics

### Task 1 Expected Outputs

1. Ranked list of marker genes correlated with LC expression
2. Statistical measures of correlation strength and significance for each marker
3. Visualizations of marker gene expression vs. LC expression
4. Report detailing the methodology used and justification for marker selection

## Task 2: Characterize Sequence Features of Consistently Expressed Genes

### Task 2 Requirements

1. **Identify Consistently Expressed Genes**
   - Define and implement criteria for "consistently expressed" genes across all experiments
   - Consider both expression level and stability across conditions
   - Apply appropriate statistical methods to identify these genes

2. **5' UTR Analysis**
   - Analyze sequence features of 5' UTRs in consistently expressed genes
   - Extract meaningful characteristics (length, GC content, secondary structures, motifs)
   - Compare with features of variably expressed genes

3. **3' UTR Analysis**
   - Analyze sequence features of 3' UTRs in consistently expressed genes
   - Extract meaningful characteristics (length, regulatory elements, polyadenylation signals)
   - Compare with features of variably expressed genes

4. **Codon Usage Analysis**
   - Calculate codon usage bias in consistently expressed genes
   - Identify preferential codon usage patterns
   - Compare with codon usage in variably expressed genes
   - Consider implications for translation efficiency

5. **Sequence Feature Integration**
   - Analyze relationships between different sequence features
   - Identify combinations of features that may contribute to consistent expression

### Task 2 Expected Outputs

1. List of consistently expressed genes across all experiments
2. Comprehensive analysis of 5' UTR, 3' UTR, and CDS sequence features in these genes
3. Comparative analysis with variably expressed genes
4. Recommendations for transgene design based on identified sequence features
5. Visualizations of key sequence characteristics

## Implementation Requirements

1. **Code Organization**
   - Implement all analysis in Python
   - Maintain modular code structure as defined in the project structure
   - Ensure appropriate separation of concerns between data loading, analysis, and visualization

2. **Documentation**
   - Document all functions with clear docstrings
   - Include explanations of methodological choices
   - Provide usage examples where appropriate

3. **Testing**
   - Create appropriate unit tests for core functionality
   - Include integration tests for end-to-end workflows
   - Ensure reproducibility of results

4. **Performance**
   - Optimize code for efficient processing of potentially large datasets
   - Consider memory usage when handling large sequence files
   - Implement appropriate caching when beneficial

## Deliverables

For each task, the following deliverables are expected:

1. **Python implementation** of all required functionality
2. **Documentation** detailing methodology and usage
3. **Visualizations** that effectively communicate findings
4. **Summary report** with conclusions and recommendations

## Success Criteria

The project will be considered successful if:

1. It identifies genes with expression levels significantly correlated with the LC product
2. It provides a ranked list of potential marker genes with clear justification
3. It characterizes sequence features of consistently expressed genes
4. It draws meaningful conclusions about sequence features that could inform future transgene design
5. All code is well-documented, tested, and follows the defined project structure.
