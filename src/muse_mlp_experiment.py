import os
import torch
import itertools
import numpy as np
import pandas as pd
import torch.optim as optim
from utils import (set_seed, load_ranked_evidence, load_evidence_features, load_ranked_verite, load_features, prepare_dataloader, eval_verite, train_step, eval_step, early_stop, save_results_csv)
from models import MUSE_MLP_CLF

def run_muse_mlp(data_path, evidence_path, verite_path, full_ablation=True, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0):
    
    num_workers=8
    epochs = 50
    early_stop_epochs = 10
    use_evidence = 1
    batch_size = 512
    seed_options = [0]
    emb_dim = 768 if '14' in encoder_version else 512    
    lr_options = [1e-4, 5e-5]    
    activation="gelu"    
    epsilon = 1e-8
    weight_decay = 0.01
    
    init_model_name = 'muse_' 
    choose_training_dataset = "_news_clippings" 
    results_filename = "results_muse" + choose_training_dataset
    data_name = 'news_clippings_balanced'
    data_name_x = 'news_clippings_balanced_external_info'

    if full_ablation:
        all_sims_to_keep = [
            ['img_txt'],
            ['img_X_img'],
            ['txt_X_img'],
            ['img_X_txt'],
            ['txt_X_txt'],
            ['X_img_X_txt'],
            ['img_X_img', 'txt_X_txt'],
            ['img_txt', 'txt_X_txt'],
            ['img_txt', 'img_X_img'],
            ['img_txt', 'img_X_img', 'txt_X_txt'],
            ['img_txt', 'img_X_img', 'txt_X_txt', 'X_img_X_txt'],    
            ['img_txt', 'img_X_img', 'txt_X_img', 'img_X_txt', 'X_img_X_txt'],    
            ['img_txt', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'],            
            ['img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'],          
            ['img_txt', 'img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'],    
        ]
    else:
        all_sims_to_keep = [
            ['img_txt', 'img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'],    
        ]        
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:" + str(choose_gpu) if torch.cuda.is_available() else "cpu")
    print(device)
    
    encoder_version = encoder_version.replace("-", "").replace("/", "")
                    
    train_data, valid_data, test_data = load_ranked_evidence(encoder, encoder_version, data_path, data_name, data_name_x)
    X_image_embeddings, X_text_embeddings = load_evidence_features(encoder, encoder_version, evidence_path, data_name, data_name_x)
    verite_test, verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings = load_ranked_verite(encoder, encoder_version, verite_path)
    image_embeddings, text_embeddings = load_features(data_path=data_path, data_name=choose_training_dataset, encoder=encoder, encoder_version=encoder_version)

    grid = itertools.product(all_sims_to_keep, lr_options, seed_options)        
    
    experiment = 0
    for params in grid:
    
        sims_to_keep, learning_rate, seed = params
    
        set_seed(seed)
    
        history = []
        has_not_improved_for = 0
    
        full_model_name = init_model_name + "_" + encoder_version + "_" + str(learning_rate) + "_" + str(emb_dim) + "_" + str(batch_size) + "_" + str(seed) + "_" + choose_training_dataset + "_" + str(use_evidence)
    
        parameters = {
                        "LEARNING_RATE": learning_rate,
                        "EPSILON": epsilon, 
                        "WEIGHT_DECAY": weight_decay,
                        "EPOCHS": epochs, 
                        "BATCH_SIZE": batch_size,
                        "USE_EVIDENCE": use_evidence,
                        "SIMS_TO_KEEP": sims_to_keep, 
                        "ACTIVATION": activation,
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
            
        model = MUSE_MLP_CLF(emb_dim, sims_to_keep)
    
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