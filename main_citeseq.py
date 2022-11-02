import argparse
import os
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from preprocess import *
from config import *
from model import *
from train import *
import time

def get_args():
    '''Add argument by command line'''
    parser = argparse.ArgumentParser()

    ######### Data arguments
    parser.add_argument('--RNA_raw_matrix', type=str,default='/data/msyuan/scMVC/data/10x10k_rawdata/rna_mat.csv', help='The path of RNA raw matrix')
    parser.add_argument('--ADT_raw_matrix', type=str,default='/data/msyuan/scMVC/data/10x10k_rawdata/prt_mat.csv', help='The path of ADT raw matrix')
    parser.add_argument('--have_labels', default=True, help='Whether have ground truth')
    parser.add_argument('--labels_path',type=str, default='/data/msyuan/scMVC/data/10x10k_rawdata/labels.csv', help='The path of labels')
    parser.add_argument('--highly_genes', type=int, default=4000)

    ######### Model arguments
    parser.add_argument('--device', type=int, default=0, help='The number of cuda device to be used')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nclusters', type=int, default=7)
    parser.add_argument('--encoder_rna_layer', type=list, default=[256,64,32])
    parser.add_argument('--encoder_adt_layer', type=list, default=[32])
    parser.add_argument('--use_bn', default=False, help='Use batch norm or not in the DDC module')
    parser.add_argument('--nhidden', type=int, default=16, help='Dimension of the hidden layer in DDC module')
    parser.add_argument('--loss_weights', type=list, default=[1.0,1.0,1.0,1.0,1.0], help='ddc_1|ddc_2|ddc_3|zinb_1|contrast')
    parser.add_argument('--rel_sigma', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='max training epoch')
    parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--model_savepath', type=str, default='/data/msyuan/scMVC/results/10x10k_model.pth')
    parser.add_argument('--results_savepath', type=str, default='/data/msyuan/scMVC/results/')
    args = parser.parse_args()
    
    return args

def main():
    args = get_args()
    device = torch.device("cuda",args.device)
    rna_mat = np.array(pd.read_csv(args.RNA_raw_matrix,index_col=0))
    prt_mat = np.array(pd.read_csv(args.ADT_raw_matrix,index_col=0))
    if args.have_labels:
        labels = np.array(pd.read_csv(args.labels_path,index_col=0), dtype=int)
        labels = np.reshape(labels, (-1))
    
        #build dataset
        views = [rna_mat, prt_mat]
        train_dataset = multiviewDataset(views, labels, device, highly_genes=[[0],[args.highly_genes]])
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            drop_last=True,
        )
        print("Building Multimodal dataset done with {} cells loaded.".format(len(train_dataset)))

        test_sampler = torch.utils.data.SequentialSampler(train_dataset)
        test_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=test_sampler,
            batch_size=len(rna_mat),
        )
        

        encoder1_cfg = encoder_cfg(Layer=tuple([np.shape((train_dataset.views[0].X))[1]]+args.encoder_rna_layer))
        encoder2_cfg = encoder_cfg(Layer=tuple([np.shape((train_dataset.views[1].X))[1]]+args.encoder_adt_layer))
        mvencodercfg = mvencoder_cfg(view1_encoder_cfg=encoder1_cfg, view2_encoder_cfg=encoder2_cfg)
        ddc_cfg = DDC_config(n_clusters=args.nclusters,n_hidden=args.nhidden,use_bn=args.use_bn,direct=True,device=args.device)
        loss_cfg = Loss_config(n_clusters=args.nclusters,device=args.device, funcs="ddc_1|ddc_2|ddc_3|zinb_1|contrast",
                               weights=args.loss_weights,
                               rel_sigma=args.rel_sigma, tau=args.tau, delta=args.delta)
        optimizer_cfg = Optimizer_config(learning_rate=args.lr)
        mvnet_cfg = scMVC_contrast_config(multiview_encoders_config=mvencodercfg,
                                          cm_config=ddc_cfg, loss_config=loss_cfg,
                                          optimizer_config=optimizer_cfg)

        # contrastive MVC
        scMVC_contrast_model = scMVC_contrast(mvnet_cfg).to(device)
        t0 = time.time()
        train_cfg(scMVC_contrast_model, train_loader, args.max_epoch, train_loader, args.batch_size, args.test_interval)
        t1 = time.time()
        trun = t1 - t0
        print("MoClust running time: %s s"%trun)
        torch.save(scMVC_contrast_model.state_dict(), args.model_savepath)
        
        labels,predictions,latent_features, fused_features, hidden_features = batch_predict(scMVC_contrast_model,test_loader, len(rna_mat), if_train=False, if_latent=True)
        labels_df = pd.DataFrame(labels)
        labels_df.to_csv(args.results_savepath + "labels.csv")
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(args.results_savepath + "pred.csv")
        latent1_df = pd.DataFrame(latent_features[0])
        latent1_df.to_csv(args.results_savepath + "latent1.csv")
        latent2_df = pd.DataFrame(latent_features[1])
        latent2_df.to_csv(args.results_savepath + "latent2.csv")
        fused_df = pd.DataFrame(fused_features)
        fused_df.to_csv(args.results_savepath + "fused.csv")
        hidden_df = pd.DataFrame(hidden_features)
        hidden_df.to_csv(args.results_savepath + "hidden.csv")
        
    else:
        views = [rna_mat, prt_mat]
        train_dataset = multiviewDataset_nolabel(views, device, highly_genes=[[0],[args.highly_genes]])
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=len(prt_mat),
        )
        
        encoder1_cfg = encoder_cfg(Layer=tuple([np.shape((train_dataset.views[0].X))[1]]+args.encoder_rna_layer))
        encoder2_cfg = encoder_cfg(Layer=tuple([np.shape((train_dataset.views[1].X))[1]]+args.encoder_adt_layer))
        mvencodercfg = mvencoder_cfg(view1_encoder_cfg=encoder1_cfg, view2_encoder_cfg=encoder2_cfg)
        ddc_cfg = DDC_config(n_clusters=args.nclusters,n_hidden=args.nhidden,use_bn=args.use_bn,direct=True,device=args.device)
        loss_cfg = Loss_config(n_clusters=args.nclusters,device=args.device, funcs="ddc_1|ddc_2|ddc_3|zinb_1|contrast",
                               weights=args.loss_weights,
                               rel_sigma=args.rel_sigma, tau=args.tau, delta=args.delta)
        optimizer_cfg = Optimizer_config(learning_rate=args.lr)
        mvnet_cfg = scMVC_contrast_config(multiview_encoders_config=mvencodercfg,
                                          cm_config=ddc_cfg, loss_config=loss_cfg,
                                          optimizer_config=optimizer_cfg)

        # contrastive MVC
        scMVC_contrast_model = scMVC_contrast(mvnet_cfg).to(device)
        t0 = time.time()
        train(scMVC_contrast_model, train_loader, args.max_epoch)
        t1 = time.time()
        trun = t1 - t0
        print("MoClust running time: %s s"%trun)
        torch.save(scMVC_contrast_model.state_dict(), args.model_savepath)
        
        predictions,latent_features, fused_features, hidden_features = batch_predict_nolabel(scMVC_contrast_model,test_loader, len(rna_mat), if_train=False, if_latent=True)
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(args.results_savepath + "pred.csv")
        latent1_df = pd.DataFrame(latent_features[0])
        latent1_df.to_csv(args.results_savepath + "latent1.csv")
        latent2_df = pd.DataFrame(latent_features[1])
        latent2_df.to_csv(args.results_savepath + "latent2.csv")
        fused_df = pd.DataFrame(fused_features)
        fused_df.to_csv(args.results_savepath + "fused.csv")
        hidden_df = pd.DataFrame(hidden_features)
        hidden_df.to_csv(args.results_savepath + "hidden.csv")
    
if __name__ == '__main__':
    main()