import os
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from ast import literal_eval
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

def str_to_list(df, list_columns):
  
    df[list_columns] = df[list_columns].fillna('[]')

    for column in list_columns:
        df[column] = df[column].apply(literal_eval)
        
    return df

def load_ranked_evidence(encoder, choose_encoder_version, data_path, data_name, data_name_X):

    encoder_version = choose_encoder_version.replace("-", "").replace("/", "")
    train_data_ranked = pd.read_csv(data_path + 'news_clippings/merged_balanced_train_ranked_' + encoder.lower() + "_" + encoder_version + '.csv', index_col=0)
    valid_data_ranked = pd.read_csv(data_path + 'news_clippings/merged_balanced_valid_ranked_' + encoder.lower() + "_" + encoder_version + '.csv', index_col=0)
    test_data_ranked = pd.read_csv(data_path + 'news_clippings/merged_balanced_test_ranked_' + encoder.lower() + "_" + encoder_version + '.csv', index_col=0)

    list_cols = ['entities', 'q_detected_labels', 'captions', 'titles', 'images_paths', 'images_labels', 'img_ranked_items', 'img_sim_scores','txt_ranked_items', 'txt_sim_scores']  
    train_data_ranked = str_to_list(train_data_ranked, list_cols)
    valid_data_ranked = str_to_list(valid_data_ranked, list_cols)
    test_data_ranked = str_to_list(test_data_ranked, list_cols)
    
    train_data_ranked[['id', 'image_id']] = train_data_ranked[['id', 'image_id']].astype('str')
    valid_data_ranked[['id', 'image_id']] = valid_data_ranked[['id', 'image_id']].astype('str')
    test_data_ranked[['id', 'image_id']] = test_data_ranked[['id', 'image_id']].astype('str') 

    train_data_ranked = train_data_ranked[train_data_ranked.id != '111288']
                    
    return train_data_ranked, valid_data_ranked, test_data_ranked

def load_evidence_features(encoder, choose_encoder_version, evidence_path, data_name, data_name_X):
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')
    
    X_image_embeddings = np.load(evidence_path + data_name_X + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_image_ids = np.load(evidence_path + data_name_X + "_image_ids_" + encoder_version +".npy")
    X_image_embeddings = pd.DataFrame(X_image_embeddings, index=X_image_ids).T
    X_image_embeddings.columns = X_image_embeddings.columns.astype('str')

    X_text_embeddings = np.load(evidence_path + data_name_X + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_text_ids = np.load(evidence_path + data_name_X + "_text_ids_" + encoder_version +".npy")

    X_text_embeddings = pd.DataFrame(X_text_embeddings, index=X_text_ids).T
    X_text_embeddings.columns = X_text_embeddings.columns.astype('str')  
    X_text_embeddings = X_text_embeddings.loc[:, ~X_text_embeddings.columns.duplicated()]
       
    return X_image_embeddings, X_text_embeddings

def load_ranked_verite(encoder, choose_encoder_version, data_path, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}):
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')
    
    verite_test = pd.read_csv(data_path + "VERITE_ranked_evidence_" + encoder.lower() + "_" + encoder_version +  ".csv", index_col=0)

    verite_test = str_to_list(verite_test, ['captions', 'images_paths', 'img_ranked_items', 'txt_ranked_items'])
    
    verite_test = verite_test.reset_index().rename({'label': 'falsified'}, axis=1)
    verite_test['image_id'] = verite_test['id']

    verite_text_embeddings = np.load(data_path + "VERITE_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = np.load(data_path + "VERITE_" + encoder.lower() +"_image_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = pd.DataFrame(verite_image_embeddings, index=verite_test.id.values).T
    verite_text_embeddings = pd.DataFrame(verite_text_embeddings, index=verite_test.id.values).T

    verite_test.falsified.replace(label_map, inplace=True)

    data_name = 'VERITE'
    X_verite_image_embeddings = np.load(data_path + data_name + '_external_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_verite_image_ids = np.load(data_path + data_name + "_external_image_ids_" + encoder_version +".npy")
    X_verite_image_embeddings = pd.DataFrame(X_verite_image_embeddings, index=X_verite_image_ids).T
    X_verite_image_embeddings.columns = X_verite_image_embeddings.columns.astype('str')

    X_verite_text_embeddings = np.load(data_path + data_name + '_external_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_verite_text_ids = np.load(data_path + data_name + "_external_text_ids_" + encoder_version +".npy")
    X_verite_text_embeddings = pd.DataFrame(X_verite_text_embeddings, index=X_verite_text_ids).T
    X_verite_text_embeddings.columns = X_verite_text_embeddings.columns.astype('str')
    
    return verite_test, verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def load_features(data_path, data_name, encoder, encoder_version, filter_ids=[None]):
    
    print("Load features")
    
    encoder_version = encoder_version.replace("-", "").replace("/", "")
    
    ocr_embeddings = None
    
    if "news_clippings" in data_name:
        image_embeddings = np.load(
            data_path + "news_clippings/news_clippings_balanced" + "_" + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy"
        ).astype("float32")

        text_embeddings = np.load(
            data_path + "news_clippings/news_clippings_balanced" + "_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy"
        ).astype("float32")

        item_ids = np.load(data_path + "news_clippings/news_clippings_balanced" + "_item_ids_" + encoder_version + ".npy")

    elif "VERITE" in data_name: 
        image_embeddings = np.load(
            data_path + data_name + "_" + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy"
        ).astype("float32")

        text_embeddings = np.load(
            data_path + data_name + "_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy"
        ).astype("float32")

        item_ids = np.load(data_path + data_name + "_item_ids_" + encoder_version + ".npy")
        
    else:
        raise("TO-DO")
        
    image_embeddings = pd.DataFrame(image_embeddings, index=item_ids).T
    text_embeddings = pd.DataFrame(text_embeddings, index=item_ids).T
    
    image_embeddings.columns = image_embeddings.columns.astype('str')
    text_embeddings.columns = text_embeddings.columns.astype('str')  
    
    if len(filter_ids) > 1:
        image_embeddings = image_embeddings[filter_ids]
        text_embeddings = text_embeddings[filter_ids]
        
    return image_embeddings, text_embeddings

def load_verite(data_path, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}):
    
    verite_test = pd.read_csv(data_path + 'VERITE.csv', index_col=0)
    verite_test = verite_test.reset_index().rename({'index': 'id', 'label': 'falsified'}, axis=1)
    verite_test['image_id'] = verite_test['id']

    verite_test.falsified.replace(label_map, inplace=True)

    verite_test.id = verite_test.id.astype("str")
    verite_test.image_id = verite_test.image_id.astype("str")

    return verite_test

def calculate_muse(input_data, split_name, image_embeddings, text_embeddings, X_image_embeddings, X_text_embeddings, out_path, use_evidence = 1):    
    all_similarities = []

    print(out_path)
    
    for i, sample in tqdm(input_data.iterrows(), total=input_data.shape[0]):    
    
        img = image_embeddings[sample.image_id].values.reshape(1, -1)
        txt = text_embeddings[sample.id].values.reshape(1, -1)
        
        X_img = X_image_embeddings[sample.img_ranked_items[:use_evidence]].T.values.reshape(1, -1)
        X_txt = X_text_embeddings[sample.txt_ranked_items[:use_evidence]].T.values.reshape(1, -1)
        
        img_txt = cosine_similarity(img, txt).item() 
    
        if X_img.shape[1] > 0:
            img_X_img = cosine_similarity(img, X_img).item()
            txt_X_img = cosine_similarity(txt, X_img).item()
        else:
            img_X_img = 0
            txt_X_img = 0
    
        if X_txt.shape[1] > 0:
            img_X_txt = cosine_similarity(img, X_txt).item()
            txt_X_txt = cosine_similarity(txt, X_txt).item()   
    
            if X_img.shape[1] > 0:
                X_img_X_txt = cosine_similarity(X_img, X_txt).item()
            else:
                X_img_X_txt = 0
        else:
            img_X_txt = 0
            txt_X_txt = 0        
            X_img_X_txt = 0
    
        d = {
            'id': sample.id,
            'image_id': sample.image_id,
            'match_index': sample.match_index,
            'X_id': sample.txt_ranked_items[:use_evidence], 
            'X_image_id': sample.img_ranked_items[:use_evidence],
            'falsified': sample.falsified,
            'img_txt': img_txt,
            'img_X_img': img_X_img,
            'txt_X_img': txt_X_img,
            'img_X_txt': img_X_txt,
            'txt_X_txt': txt_X_txt,
            'X_img_X_txt': X_img_X_txt,
        }
    
        all_similarities.append(d)
    
    pd.DataFrame(all_similarities).to_csv(out_path + ".csv")



def check_C(C, pos):
    
    if C == 0:
        return np.zeros(pos.shape[0])    
    else: 
        return np.ones(pos.shape[0])
        
        
def sensitivity_per_class(y_true, y_pred, C):
    
    pos = np.where(y_true == C)[0]
    y_true = y_true[pos]
    y_pred = y_pred[pos]
    
    if C == 2:
        y_true = np.ones(y_true.shape[0]).reshape(-1, 1)
    
    return round((y_pred == y_true).sum() / y_true.shape[0], 4)

def accuracy_CvC(y_true, y_pred, Ca, Cb):
    pos_a = np.where(y_true == Ca)[0]
    pos_b = np.where(y_true == Cb)[0]

    y_pred_a = y_pred[pos_a].flatten()
    y_pred_b = y_pred[pos_b].flatten()   
    
    y_true_a = check_C(Ca, pos_a)
    y_true_b = check_C(Cb, pos_b)
    
    y_pred_avb = np.concatenate([y_pred_a, y_pred_b])
    y_true_avb = np.concatenate([y_true_a, y_true_b])
    
    return round(metrics.accuracy_score(y_true_avb, y_pred_avb), 4)

class DatasetIterator(torch.utils.data.Dataset):
    
    def __init__(
        self,
        input_data,
        visual_features,
        textual_features,
        X_visual_features,
        X_textual_features,
        use_evidence=False,
    ):
        self.input_data = input_data
        self.visual_features = visual_features
        self.textual_features = textual_features
        
        self.X_visual_features = X_visual_features
        self.X_textual_features = X_textual_features
        
        self.use_evidence = use_evidence
        
    def __len__(self):
        return self.input_data.shape[0]
        
    def __getitem__(self, idx):
        
        current = self.input_data.iloc[idx]
        
        img = self.visual_features[current.image_id].values.astype("float32")
        txt = self.textual_features[current.id].values.astype("float32") 

        label = float(current.falsified)
                
        if self.use_evidence:
            X_img = self.X_visual_features[current.img_ranked_items[:self.use_evidence]].T.values.astype("float32")
            X_txt = self.X_textual_features[current.txt_ranked_items[:self.use_evidence]].T.values.astype("float32")
            
            if X_img.shape[0] < self.use_evidence:            
                pad_zeros = np.zeros((self.use_evidence - X_img.shape[0], self.X_visual_features.shape[0]))
                X_img = np.vstack([X_img, pad_zeros.astype("float32")])

            if X_txt.shape[0] < self.use_evidence:
                pad_zeros = np.zeros((self.use_evidence - X_txt.shape[0], self.X_textual_features.shape[0]))
                X_txt = np.vstack([X_txt, pad_zeros.astype("float32")])       
                
        else:
            X_img = np.array(np.nan)
            X_txt = np.array(np.nan) 
        
        return img, txt, label, X_img, X_txt
         
def prepare_dataloader(input_data, 
                       visual_features,
                       textual_features, 
                       X_visual_features,
                       X_textual_features,   
                       batch_size, 
                       num_workers, 
                       shuffle, 
                       use_evidence=True,
                      ):
    
    dg = DatasetIterator(
        input_data,
        visual_features,
        textual_features,
        X_visual_features,
        X_textual_features,        
        use_evidence
    )

    dataloader = DataLoader(
        dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader


def eval_verite(model, verite_data, verite_dataloader, device, label_map={'true': 0, 'miscaptioned': 1, 'out-of-context': 2}, cur_epoch=-3):
    
    print("\nEvaluation on VERITE")
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for i, data in enumerate(verite_dataloader, 0):

            images = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True)
            labels = data[2]
            
            images_X = data[3].to(device, non_blocking=True)
            texts_X = data[4].to(device, non_blocking=True)     

            output_clf = model(images, texts, images_X, texts_X)

            y_pred.extend(output_clf.detach().cpu().numpy())
            y_true.extend(labels.detach().cpu().numpy())

    y_true = np.array(y_true).reshape(-1,1)
    y_pred = np.vstack(y_pred)
    y_pred = 1/(1 + np.exp(-y_pred))

    y_pred = y_pred.round()
    
    verite_results = {}

    verite_results['True'] = sensitivity_per_class(y_true, y_pred, 0)
    verite_results['Miscaptioned'] = sensitivity_per_class(y_true, y_pred, 1)
    verite_results['Out-Of-Context'] = sensitivity_per_class(y_true, y_pred, 2)

    verite_results['true_v_miscaptioned'] = accuracy_CvC(y_true, y_pred, 0, 1)
    verite_results['true_v_ooc'] = accuracy_CvC(y_true, y_pred, 0, 2)
    verite_results['miscaptioned_v_ooc'] = accuracy_CvC(y_true, y_pred, 1, 2)

    y_true_all = y_true.copy()
    y_true_all[np.where(y_true_all == 2)[0]] = 1

    verite_results['accuracy'] = round(metrics.accuracy_score(y_true_all, y_pred), 4)
    verite_results['balanced_accuracy'] = round(metrics.balanced_accuracy_score(y_true_all, y_pred), 4)

    print(verite_results)

    return verite_results


def train_step(model, input_dataloader, current_epoch, optimizer, device, batches_per_epoch):
    
    epoch_start_time = time.time()

    running_loss = 0.0
    
    model.train()
    
    for i, data in enumerate(input_dataloader, 0):

        images = data[0].to(device, non_blocking=True)
        texts = data[1].to(device, non_blocking=True).squeeze(1)
        labels = data[2].to(device, non_blocking=True)
        images_X = data[3].to(device, non_blocking=True)
        texts_X = data[4].to(device, non_blocking=True)
        
        optimizer.zero_grad()
                
        output_clf = model(images, texts, images_X, texts_X)
                            
        loss = F.binary_cross_entropy_with_logits(output_clf.float(), labels.float())            
                
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        print(
            f"[Epoch:{current_epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}.",
            end="\r",
        )     
        
def eval_step(model, input_dataloader, current_epoch, device, batches_per_epoch):
    
    epoch_start_time = time.time()

    running_loss = 0.0
    
    y_true = []
    y_pred = []
        
    model.eval()
    
    with torch.no_grad():
    
        for i, data in enumerate(input_dataloader, 0):

            images = data[0].to(device, non_blocking=True)
            texts = data[1].to(device, non_blocking=True).squeeze(1)
            labels = data[2].to(device, non_blocking=True)            
            images_X = data[3].to(device, non_blocking=True)
            texts_X = data[4].to(device, non_blocking=True)
            
            output_clf = model(images, texts, images_X, texts_X)
            
            loss = F.binary_cross_entropy_with_logits(output_clf.float(), labels.float())                                      
                                
            y_pred.extend(output_clf.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())

            running_loss += loss.item()
            
            print(
                f"[Epoch:{current_epoch + 1}, Batch:{i + 1:5d}/{batches_per_epoch}]. Passed time: {round((time.time() - epoch_start_time) / 60, 1)} minutes. loss: {running_loss / (i+1):.3f}",
                end="\r",
            )    
            
    final_loss = running_loss / (i+1)
                    
    y_pred = np.vstack(y_pred)
    y_pred = 1/(1 + np.exp(-y_pred))
    y_true = np.array(y_true).reshape(-1,1)
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    y_pred = np.round(y_pred)        
    acc = metrics.accuracy_score(y_true, y_pred)    
    prec = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred) 
    f1 = metrics.f1_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred, normalize="true").diagonal()
    
    results = {
        "epoch": current_epoch,
        "loss": round(final_loss, 4),
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        'Pristine': round(cm[0], 4),
        'Falsified': round(cm[1], 4),
    }
    print(results)
    
    return results

def topsis(xM, wV=None):
    m, n = xM.shape

    if wV is None:
        wV = np.ones((1, n)) / n
    else:
        wV = wV / np.sum(wV)

    normal = np.sqrt(np.sum(xM**2, axis=0))

    rM = xM / normal
    tM = rM * wV
    twV = np.max(tM, axis=0)
    tbV = np.min(tM, axis=0)
    dwV = np.sqrt(np.sum((tM - twV) ** 2, axis=1))
    dbV = np.sqrt(np.sum((tM - tbV) ** 2, axis=1))
    swV = dwV / (dwV + dbV)

    arg_sw = np.argsort(swV)[::-1]

    r_sw = swV[arg_sw]

    return np.argsort(swV)[::-1]

def choose_best_model(input_df, metrics, epsilon=1e-6):

    X0 = input_df.copy()
    X0 = X0.reset_index(drop=True)
    X1 = X0[metrics]
    X1 = X1.reset_index(drop=True)
    
    X1[:-1] = X1[:-1] + epsilon 
    
    if "Accuracy" in metrics:
        X1["Accuracy"] = 1 - X1["Accuracy"]    

    if "Precision" in metrics:
        X1["Precision"] = 1 - X1["Precision"]    

    if "Recall" in metrics:
        X1["Recall"] = 1 - X1["Recall"]          
        
    if "AUC" in metrics:
        X1["AUC"] = 1 - X1["AUC"]
        
    if "F1" in metrics:
        X1["F1"] = 1 - X1["F1"]

    if "Pristine" in metrics:
        X1["Pristine"] = 1 - X1["Pristine"]
        
    if "Falsified" in metrics:
        X1["Falsified"] = 1 - X1["Falsified"]
        
    X_np = X1.to_numpy()
    best_results = topsis(X_np)
    top_K_results = best_results[:1]
    return X0.iloc[top_K_results]

def early_stop(has_not_improved_for, model, optimizer, history, current_epoch, PATH, metrics_list):

    best_index = choose_best_model(
        pd.DataFrame(history), metrics=metrics_list
    ).index[0]
        
    if not os.path.isdir(PATH.split('/')[0]):
        os.mkdir(PATH.split('/')[0])

    if current_epoch == best_index:
        
        print("Checkpoint!\n")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            PATH,
        )

        has_not_improved_for = 0
    else:
        
        print("DID NOT CHECKPOINT!\n")
        has_not_improved_for += 1
            
    return has_not_improved_for

def save_results_csv(output_folder_, output_file_, model_performance_):
    print("Save Results ", end=" ... ")
    exp_results_pd = pd.DataFrame(pd.Series(model_performance_)).transpose()
    if not os.path.isfile(output_folder_ + "/" + output_file_ + ".csv"):
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            header=True,
            index=False,
            columns=list(model_performance_.keys()),
        )
    else:
        exp_results_pd.to_csv(
            output_folder_ + "/" + output_file_ + ".csv",
            mode="a",
            header=False,
            index=False,
            columns=list(model_performance_.keys()),
        )
    print("Done\n")
