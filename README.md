# MoClust
A pytorch implement of single-cell multi-omics clustering method MoClust.

![image](https://user-images.githubusercontent.com/52522175/160607926-0e77a9b5-7a1a-4b88-8f64-299de3029092.png)


# Abstract
 Single-cell multiomics sequencing techniques have rapidly developed in the past few years. Analyzing single-cell multiomics data may give us novel perspectives to dissect cellular heterogeneity, yet integrative analysis remains challenging. The inherited high-dimensional and highly sparse omics data making it a great difficulty to reduce the dimension of each omic data. And existing integration methods are mostly stumped by aligning the omic-specific latent features and obtaining a cell state representation well suited for clustering.
 
We present MoClust, a novel joint clustering methods that can be applied to several types of single-cell multiomics data. Introducing a contrastive learning based alignment technique, MoClust is able to to learn common representations that well suited for clustering, while simultaneously considering the topology structure of latent features. Furthermore,we proposed a novel automatic doublet discovery module that can efficiently find doublets without manually setting a threshold. Extensive experiments demonstrated the powerful alignment and clustering ability of MoClust.

# Environment
python >= 3.7

- scanpy == 1.6.0
- numpy == 1.21.6
- pandas == 1.3.5
- torch == 1.10.2
- scikit-learn == 1.0.2
- scipy == 1.4.1
- seaborn == 0.9.0
- tabulate = 0.8.9
- typing == 3.5.0
- pydantic == 1.10.2   

# Data Format
Before we get started, we need to preprocess your CITE-seq or SNARE-seq data 

    - RNA data -- a cell x gene csv file
    - Protein data -- a cell x protein csv file
    - ATAC data --  a cell x peak csv file
        - the columns of ATAC file should be like chr1:56782095-56782395
        
A gtf file compatible with your data is also needed when training MoClust over SNARE-seq data

# Train MoClust over Multi-Omics data
We provide an example CITE-seq data with ground truth labels, you can train MoClust over it by

    python main_citeseq --RNA_raw_matrix='/rna_mat.csv' --ADT_raw_matrix='/prt_mat.csv -- have_labels=True --labels_path='/labels.csv'

You can train MoClust over un-annotated CITE-seq data by

    python main_citeseq --RNA_raw_matrix='/rna.csv' --ADT_raw_matrix='/adt.csv
    
You can train MoClust over un-annotated SNARE-seq data by

    python main_snareseq --RNA_raw_matrix='/rna.csv' --ATAC_raw_matrix='/atac.csv --gtf='/gencode.v39.annotation.gtf'
    
# Parameters of Moclust
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
>
>- device: the number of cuda device to be used
>
>- model_savepath: the path of the pth file to save the trained model
>
>- results_savepath: the path of a folder to save results


MoClust Model Parameters:

>- nclusters: the number of clusters
>
>- encoder_rna_layer: the dimensions of hidden layers of RNA encoder, default as [256,64,32]
>
>- encoder_adt_layer: the dimensions of hidden layers of protein encoder, default as [32]
>
>- use_bn: Use batch norm or not in the DDC module
>
>- nhidden: the dimension of the hidden layer in DDC module, default as 16

Training settings:

>- batch_size:default as 256
>
>- lr: learning rate, default as 1e-3
>
>- max_epoch: max training epoch, default as 200
>
>- test_interval: test frequency, default as 10

Hyper-parameters:

>- loss_weights: the weights of loss terms ddc_1|ddc_2|ddc_3|zinb_1|contrast, default as [1.0,1.0,1.0,1.0,1.0]
>
>- rel_sigma: sigma value used when calculating similarity matrix K in Eq (9)(10), default as 0.1
>
> - tau: tau value used when calculating cosine similarity between latent representations in Eq (6), default as 0.1
> 
> - delta: constrains the strength of contrastive loss in Eq (13), default as 0.1

# Quick Install
Package MoClust can be directly downloaded by

    conda create -n MoClust python=3.8
    conda activate MoClust
    pip install MoClust==0.0.3
 
 
# Train MoClust over Multi-Omics data
We provide an example to apply MoClust over CITE-seq data using package MoClust

    import MoClust
    import torch

    device = torch.device("cuda",0)
    views = [rna_mat, prt_mat]
    train_dataset = MoClust.multiviewDataset(views,labels, device ,highly_genes=[[0],[2000]])
    sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=256,
        drop_last=True,
    )
    print("Building Multimodal dataset done with {} cells loaded.".format(len(train_dataset)))  

    import time

    encoder1_cfg = MoClust.encoder_cfg(Layer=(np.shape(train_dataset.views[0].X)[1],256,64,32))
    encoder2_cfg = MoClust.encoder_cfg(Layer=(np.shape(train_dataset.views[1].X)[1],32))
    mvencoder_cfg = MoClust.mvencoder_cfg(view1_encoder_cfg=encoder1_cfg, view2_encoder_cfg=encoder2_cfg)
    ddc_cfg = MoClust.DDC_config(n_clusters=6, n_hidden=16,device=0)
    loss_cfg = MoClust.Loss_config(n_clusters=6,device=0, funcs="ddc_1|ddc_2|ddc_3|contrast",
                                   weights=[1.0,1.0,1.0,0.05],
                                   rel_sigma=0.15, tau=0.1, delta=0.01, gamma=1.0)
    optimizer_cfg = MoClust.Optimizer_config(learning_rate=0.0001)
    mvnet_cfg = MoClust.scMVC_contrast_config(multiview_encoders_config=mvencoder_cfg,
                                              cm_config=ddc_cfg, loss_config=loss_cfg,
                                              optimizer_config=optimizer_cfg)

    scMVC_contrast_model = MoClust.scMVC_contrast(mvnet_cfg).to(device)
    t0 = time.time()
    MoClust.train_cfg(scMVC_contrast_model, train_loader, 100, train_loader, 256, 10)
    t1 = time.time()
    trun = t1 - t0
    print("MoClust running time: %s s"%trun)
    
    
