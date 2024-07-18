import os
import torch
import itertools
import numpy as np
import pandas as pd
import torch.optim as optim
from models import AITR
from utils import (set_seed, load_ranked_evidence, load_evidence_features, load_ranked_verite, load_features, prepare_dataloader, eval_verite, train_step, eval_step, early_stop, save_results_csv)

def run_aitr(data_path, 
             evidence_path, 
             verite_path, 
             encoder = 'CLIP', 
             encoder_version = 'ViT-L/14', 
             choose_gpu = 0, 
             use_muse = True, 
             use_evidence = 1, 
             transformer_version = "aitr", 
             pooling_method = "attention_pooling"):
    
    num_workers=8
    epochs = 50
    early_stop_epochs = 10
    batch_size = 512
    seed_options = [0]
    activation="gelu"    
    dropout=0.1
    epsilon = 1e-8
    weight_decay = 0.01
    lr_options = [1e-4, 5e-5]    
    init_model_name = 'muse_' 
    choose_training_dataset = "_news_clippings" 
    results_filename = "results_muse" + choose_training_dataset
    data_name = 'news_clippings_balanced'
    data_name_x = 'news_clippings_balanced_external_info'
    lr_options = [1e-4, 5e-5]
    tf_dim_options = [256, 1024, 2048]    
    if transformer_version == "default":
        
        tf_heads_layers = [
            [4, 4, 4, 4],
            [8, 8, 8, 8],
        ] 
        
    elif transformer_version == "aitr":
        tf_heads_layers = [
            [1, 2, 4, 8],
            [8, 4, 2, 1], 
            [4, 4, 4, 4],          
            [8, 8, 8, 8],                   
        ] 
        
    else:
        raise Exception("Choose either transformer_version = 'default' or 'aitr'")    

    emb_dim = 768 if '14' in encoder_version else 512        
    
    if transformer_version == "aitr" and pooling_method == None:
        raise Exception("Choose pooling method for AITR")
        
    if transformer_version == "default" and pooling_method != None:
        raise Exception("Remove pooling method for Tranformer:default")
    
    init_model_name = 'aitr_' 
    choose_training_dataset = "_news_clippings"
    results_filename = "results_aitr" + choose_training_dataset
    
    data_name = 'news_clippings_balanced'
    data_name_x = 'news_clippings_balanced_external_info'
    
    fusion_method = ["concat_1", "add", "sub", "mul"]
    fuse_evidence =["concat_1"] if use_evidence else [False]
        
    if use_muse == True:
        if use_evidence:
            sims_to_keep = ['img_txt', 'img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'] 
        else:
            sims_to_keep = ['img_txt']           
    else:
        sims_to_keep = [] 
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")
    print(device)
    
    encoder_version = encoder_version.replace("-", "").replace("/", "") #.lower() 
    
    train_data, valid_data, test_data = load_ranked_evidence(encoder, encoder_version, data_path, data_name, data_name_x)
    X_image_embeddings, X_text_embeddings = load_evidence_features(encoder, encoder_version, evidence_path, data_name, data_name_x)
    verite_test, verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings = load_ranked_verite(encoder, encoder_version, verite_path)
    image_embeddings, text_embeddings = load_features(data_path=data_path, data_name=choose_training_dataset, encoder=encoder, encoder_version=encoder_version)

    grid = itertools.product(lr_options, tf_heads_layers, tf_dim_options, seed_options)        
    
    experiment = 0
    for params in grid:
    
        learning_rate, tf_h_l, tf_dim, seed = params
    
        set_seed(seed)
    
        history = []
        has_not_improved_for = 0
    
        full_model_name = init_model_name + "_" + encoder_version + "_" + str(learning_rate) + "_" + str(emb_dim) + "_" + str(batch_size) + "_" + str(seed) + "_" + choose_training_dataset + "_" + str(use_evidence) + "_" + str(use_muse) + "_" + str(transformer_version) + "_" + str(pooling_method)
    
        parameters = {
                        "LEARNING_RATE": learning_rate,
                        "EPSILON": epsilon, 
                        "WEIGHT_DECAY": weight_decay,
                        "EPOCHS": epochs, 
                        "BATCH_SIZE": batch_size,
                        "TRANSFORMER_VERSION": str(transformer_version),
                        "POOLING_MECHANISM": str(pooling_method), 
                        "USE_EVIDENCE": use_evidence,
                        "FUSE_EVIDENCE_METHOD": fuse_evidence,
                        "TF_H_L": tf_h_l,
                        "TF_DIM": tf_dim,
                        "USE_MUSE" : use_muse, 
                        "SIMS_TO_KEEP": sims_to_keep, 
                        "NUM_WORKERS": num_workers,
                        "USE_FEATURES": ["images", "texts"],
                        "EARLY_STOP_EPOCHS": early_stop_epochs,
                        "ENCODER": encoder,
                        "ENCODER_VERSION": encoder_version,
                        "SEED": seed,
                        "full_model_name": full_model_name,
                    }
        
        PATH = "checkpoints_pt/model_" + full_model_name + ".pt"  
    
        train_dataloader = prepare_dataloader(input_data=train_data,
                                              visual_features=image_embeddings,
                                              textual_features=text_embeddings,
                                              X_visual_features=X_image_embeddings,
                                              X_textual_features=X_text_embeddings,
                                              batch_size=batch_size, 
                                              num_workers=num_workers, 
                                              shuffle=True,
                                              use_evidence=use_evidence
                                             )
    
        valid_dataloader = prepare_dataloader(input_data=valid_data,
                                              visual_features=image_embeddings,
                                              textual_features=text_embeddings,
                                              X_visual_features=X_image_embeddings,
                                              X_textual_features=X_text_embeddings,   
                                              batch_size=batch_size, 
                                              num_workers=num_workers, 
                                              shuffle=False,
                                              use_evidence=use_evidence
                                             )
        
        test_dataloader = prepare_dataloader(input_data=test_data,
                                              visual_features=image_embeddings,
                                              textual_features=text_embeddings, 
                                              X_visual_features=X_image_embeddings,
                                              X_textual_features=X_text_embeddings,      
                                              batch_size=batch_size, 
                                              num_workers=num_workers, 
                                              shuffle=False,
                                              use_evidence=use_evidence                                        
                                            )
    
        verite_dataloader = prepare_dataloader(input_data=verite_test,
                                               visual_features=verite_image_embeddings,
                                               textual_features=verite_text_embeddings,  
                                               X_visual_features=X_verite_image_embeddings,
                                               X_textual_features=X_verite_text_embeddings,   
                                               batch_size=batch_size, 
                                               num_workers=num_workers, 
                                               shuffle=False,
                                               use_evidence=use_evidence)
            
        model = AITR(emb_dim, 
                     fusion_method, 
                     use_evidence, 
                     use_muse, 
                     sims_to_keep, 
                     transformer_version,
                     tf_h_l,
                     tf_dim,
                     pooling_method
                    )
    
        model.to(device)
        
        print(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon, weight_decay=weight_decay)
                
        for current_epoch in range(epochs):
            
            print("Path:", PATH)
            
            train_step(model, 
                       train_dataloader, 
                       current_epoch, 
                       optimizer, 
                       device, 
                       batches_per_epoch=train_dataloader.__len__())
                    
            results = eval_step(model, 
                                valid_dataloader, 
                                current_epoch,  
                                device, 
                                batches_per_epoch=valid_dataloader.__len__())   
            
            history.append(results)
                
            has_not_improved_for = early_stop(
                has_not_improved_for,
                model,
                optimizer,
                history,
                current_epoch,
                PATH,
                metrics_list=["Accuracy"],
            )
        
            if has_not_improved_for >= early_stop_epochs:
        
                print(
                    f"Performance has not improved for {early_stop_epochs} epochs. Stop training at epoch {current_epoch}!"
                )
                break
                
        print("Finished Training. Loading the best model from checkpoints.")
        
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
            
        res_val = eval_step(model, 
                            valid_dataloader, 
                            -1, 
                            device, 
                            batches_per_epoch=valid_dataloader.__len__())
    
        res_test = eval_step(model, 
                             test_dataloader, 
                             -1, 
                             device, 
                             batches_per_epoch=test_dataloader.__len__())
        
        
        verite_results = eval_verite(model, verite_test, verite_dataloader, device)
            
        res_verite = {
            "verite_" + str(key.lower()): val for key, val in verite_results.items()
        }
    
        res_val = {
            "valid_" + str(key.lower()): val for key, val in res_val.items()
        }
        
        res_test = {
            "test_" + str(key.lower()): val for key, val in res_test.items()
        }
        
        all_results = {**parameters, **res_test, **res_val, **res_verite}
        
        all_results["path"] = PATH
        all_results["history"] = history
        
        if not os.path.isdir("results"):
            os.mkdir("results")
        
        save_results_csv(
            "results/",
            results_filename,            
            all_results,
        )
        
    
