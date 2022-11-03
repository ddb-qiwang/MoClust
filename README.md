# MoClust
A pytorch implement of single-cell multiomic integrating method MoClust.

![image](https://user-images.githubusercontent.com/52522175/160607926-0e77a9b5-7a1a-4b88-8f64-299de3029092.png)


# Abstract
 Single-cell multiomics sequencing techniques have rapidly developed in the past few years. Analyzing single-cell multiomics data may give us novel perspectives to dissect cellular heterogeneity, yet integrative analysis remains challenging. The inherited high-dimensional and highly sparse omics data making it a great difficulty to reduce the dimension of each omic data. And existing integration methods are mostly stumped by aligning the omic-specific latent features and obtaining a cell state representation well suited for clustering.
 
We present MoClust, a novel joint clustering methods that can be applied to several types of single-cell multiomics data. Introducing a contrastive learning based alignment technique, MoClust is able to to learn common representations that well suited for clustering, while simultaneously considering the topology structure of latent features. Furthermore,we proposed a novel automatic doublet discovery module that can efficiently find doublets without manually setting a threshold. Extensive experiments demonstrated the powerful alignment and clustering ability of MoClust.

# Data Format
Before we get started, we need to preprocess your CITE-seq or SNARE-seq data 

    RNA data -- a cell x gene csv file
    Protein data -- a cell x protein csv file
    ATAC data --  a cell x peak csv file
        the columns of ATAC file should be like chr1:56782095-56782395
        
A gtf file compatible with your data is also needed when training MoClust over SNARE-seq data

# Training
You can train MoClust over CITE-seq data by

    python main_citeseq --RNA_raw_matrix='/rna.csv' --ADT_raw_matrix='/adt.csv
    
You can train MoClust over SNARE-seq data by

    python main_snareseq --RNA_raw_matrix='/rna.csv' --ATAC_raw_matrix='/atac.csv --gtf='/gencode.v39.annotation.gtf'
    
# Hyper-parameters
The list of parameters is given below:

>- RNA_raw_matrix: the path of rna matrix csv file
>
>- ADT_raw_matrix: the path of protein matrix csv file
>
>- have_labels: have ground truth or not
>
>- labels_path: the path of ground truth csv file
>
>- highly_genes: the number of highly variable genes to be selected



