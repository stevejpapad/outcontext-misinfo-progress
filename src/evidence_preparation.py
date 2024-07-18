import os
import json
import torch
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from string import digits
from ast import literal_eval
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def load_visual_news(data_path):
    
    vn_data = json.load(open(data_path + 'VisualNews/origin/data.json'))
    vn_data = pd.DataFrame(vn_data)
    vn_data['image_id'] = vn_data['id']
        
    return vn_data

def load_features(data_path, data_name, encoder, encoder_version, filter_ids=[None]):
    
    print("Load features")
    
    encoder_version = encoder_version.replace("-", "").replace("/", "")
        
    image_embeddings = np.load(
        data_path + "news_clippings/" + data_name + "_" + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy"
    ).astype("float32")

    text_embeddings = np.load(
        data_path + "news_clippings/" + data_name + "_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy"
    ).astype("float32")

    item_ids = np.load(data_path + "news_clippings/" + data_name + "_item_ids_" + encoder_version + ".npy")
                       
    image_embeddings = pd.DataFrame(image_embeddings, index=item_ids).T
    text_embeddings = pd.DataFrame(text_embeddings, index=item_ids).T
    
    image_embeddings.columns = image_embeddings.columns.astype('str')
    text_embeddings.columns = text_embeddings.columns.astype('str')  
    
    if len(filter_ids) > 1:
        image_embeddings = image_embeddings[filter_ids]
        text_embeddings = text_embeddings[filter_ids]
        
    return image_embeddings, text_embeddings

def load_training_dataset(data_path, data_name):
        
        
    print("Data path: ", data_path)
    print("Load: ", data_name)
        
    train_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/train.json"))
    valid_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/val.json"))
    test_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/test.json"))

    train_data = pd.DataFrame(train_data["annotations"])
    valid_data = pd.DataFrame(valid_data["annotations"])
    test_data = pd.DataFrame(test_data["annotations"])
                        
    train_data.id = train_data.id.astype('str')
    valid_data.id = valid_data.id.astype('str')
    test_data.id = test_data.id.astype('str')
    
    train_data.image_id = train_data.image_id.astype('str')
    valid_data.image_id = valid_data.image_id.astype('str')
    test_data.image_id = test_data.image_id.astype('str')       

    train_data = train_data[train_data.id != '111288']    
    train_data = train_data[train_data.image_id != '111288']    

    return train_data, valid_data, test_data

def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    input_str = unidecode(input_str)  
    return input_str

def fetch_evidence_split(evidence_path):

    train_paths = pd.DataFrame(json.load(open(evidence_path + 'dataset_items_train.json'))).transpose()
    valid_paths = pd.DataFrame(json.load(open(evidence_path + 'dataset_items_val.json'))).transpose()
    test_paths = pd.DataFrame(json.load(open(evidence_path + 'dataset_items_test.json'))).transpose()

    train_paths = train_paths.reset_index().rename(columns={'index': 'match_index'})
    valid_paths = valid_paths.reset_index().rename(columns={'index': 'match_index'})
    test_paths = test_paths.reset_index().rename(columns={'index': 'match_index'})
    
    return train_paths, valid_paths, test_paths

def remove_duplicates_preserve_order(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def load_evidence_file(input_data, evidence_path):
    
    count_captions = 0
    all_samples = []

    for (row) in tqdm(input_data.itertuples(), total=input_data.shape[0]):

        texts_path = evidence_path + 'merged_balanced' + '/' + row.inv_path
        images_path = evidence_path + 'merged_balanced' + '/' + row.direct_path

        inv_annotation = json.load(open(texts_path + '/inverse_annotation.json'))
        q_detected_labels = json.load(open(texts_path + '/query_detected_labels'))

        img_detected_labels = json.load(open(images_path + '/detected_labels'))
        direct_annotation = json.load(open(images_path + '/direct_annotation.json'))

        images_keep_ids = json.load(open(images_path + '/imgs_to_keep_idx'))['index_of_images_tokeep']
        all_image_paths = [images_path + '/' + x + '.jpg' for x in images_keep_ids]

        titles = []

        if 'all_fully_matched_captions' in inv_annotation.keys():    
            titles = [x.get('title', '') for x in inv_annotation['all_fully_matched_captions']]


        if os.path.isfile(texts_path + '/captions_info'):

            count_captions += 1

            caption = json.load(open(texts_path + '/captions_info'))
            caption_keep_ids = json.load(open(texts_path + '/captions_to_keep_idx'))
            keep_captions = [caption['captions'][x] for x in caption_keep_ids['index_of_captions_tokeep']]

        else:
            keep_captions = []
    
        
        processed_list = [process_string(x) for x in keep_captions]
        keep_captions = remove_duplicates_preserve_order(keep_captions) # processed_list
        
        sample = {
            'match_index': row.match_index,
            'entities': inv_annotation['entities'],
            'entities_len': len(inv_annotation['entities']),
            'q_detected_labels': q_detected_labels['labels'],
            'q_detected_labels_len': len(q_detected_labels['labels']),
            'captions': keep_captions,
            'len_captions': len(keep_captions),
            'titles': titles,
            'titles_len': len(titles),
            'images_paths': all_image_paths,
            'len_images': len(all_image_paths),
            'images_labels': [img_detected_labels[img_id]['labels'] for img_id in images_keep_ids]
        }

        all_samples.append(sample)

    return all_samples

def load_merge_evidence_w_data(input_data, data_paths, evidence_path):
    
    data_evidence = load_evidence_file(data_paths, evidence_path)
    data_evidence = pd.DataFrame(data_evidence)

    data_evidence.match_index = data_evidence.match_index.astype('int')
    data_merge = pd.merge(data_evidence, input_data, on='match_index', how='right')
    
    return data_merge


def str_to_list(df, list_columns):
  
    df[list_columns] = df[list_columns].fillna('[]')

    for column in list_columns:
        df[column] = df[column].apply(literal_eval)
        
    return df

def load_merge_evidence_data(evidence_path, data_path, data_name):

    print("Load evidence paths")
    train_paths, valid_paths, test_paths = fetch_evidence_split(evidence_path)
    
    print("Load", data_name)
    train_data, valid_data, test_data = load_training_dataset(data_path, data_name)

    train_data = train_data.reset_index().rename(columns={'index': 'match_index'})
    valid_data = valid_data.reset_index().rename(columns={'index': 'match_index'})
    test_data = test_data.reset_index().rename(columns={'index': 'match_index'})

    print("Prepare train - evidence")
    train_merge = load_merge_evidence_w_data(train_data, train_paths, evidence_path)
    print("Save data+evidence", data_path + 'news_clippings/merged_balanced_train.csv')
    train_merge.to_csv(data_path + 'news_clippings/merged_balanced_train.csv')  

    print("Prepare valid - evidence")
    valid_merge = load_merge_evidence_w_data(valid_data, valid_paths, evidence_path)
    print("Save data+evidence", data_path + 'news_clippings/merged_balanced_valid.csv')
    valid_merge.to_csv(data_path + 'news_clippings/merged_balanced_valid.csv')
    
    print("Prepare test - evidence")
    test_merge = load_merge_evidence_w_data(test_data, test_paths, evidence_path)
    print("Save data+evidence", data_path + 'news_clippings/merged_balanced_test.csv')        
    test_merge.to_csv(data_path + 'news_clippings/merged_balanced_test.csv')




def idx_captions(df, prefix):
    
    captions_tuples = []
    
    for idx, caption_list in zip(df['match_index'], df['captions']):

        if isinstance(caption_list, list) and len(caption_list) > 0:
                        
            for i in range(len(caption_list)):
            
                captions_tuples.append((idx, idx + '_' + str(i), caption_list[i]))

        else:
            captions_tuples.append((idx, '-1', ''))     
        
    return pd.DataFrame(captions_tuples, columns=['match_index', 'X_item_index', 'X_caption'])

def idx_images(df, prefix):
    
    images_tuples = [] 
    
    for idx, images_list in zip(df['match_index'], df['images_paths']):

        if isinstance(images_list, list) and len(images_list) > 0:
            
            for i in range(len(images_list)):
            
                images_tuples.append((idx, idx + '_' + str(i), images_list[i]))
            
        else:
            images_tuples.append((idx, idx + '_0', ''))       
        
    return pd.DataFrame(images_tuples, columns=['match_index', 'X_item_index', 'X_image_path'])


class ImageIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        vis_processors,
        encoder_version
    ):
        self.input_data = input_data
        self.vis_processors = vis_processors
        self.encoder_version = encoder_version

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        
        img_path = current.X_image_path
        idx = current.X_item_index
                
        try:
            
            image = Image.open(img_path)
            image = image.convert('RGB')
        
            max_size = 400
            width, height = image.size

            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            image = image.resize((new_width, new_height))
            img = self.vis_processors(image)   

            return idx, img
        
        except Exception as e:
            print(e)
            print(idx)
    
class TextIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        txt_processors,
        encoder_version
    ):
        self.input_data = input_data
        self.txt_processors = txt_processors
        self.encoder_version = encoder_version

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]
        txt = self.txt_processors(current.X_caption)                       
        idx = current.X_item_index

        return idx, txt
    
def prepare_source_dataloaders(image_data, text_data, vis_processors, txt_processors, encoder_version, batch_size, num_workers, shuffle):

    img_dg = ImageIteratorSource(image_data,  vis_processors, encoder_version)

    img_dataloader = DataLoader(
        img_dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    txt_dg = TextIteratorSource(text_data,  txt_processors, encoder_version)

    txt_dataloader = DataLoader(
        txt_dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return img_dataloader, txt_dataloader


def extract_features_for_evidence(data_path, 
                                  output_path, 
                                  data_name_X, 
                                  encoder='CLIP', 
                                  choose_encoder_version='ViT-L/14', 
                                  choose_gpu=0, 
                                  batch_size=256, 
                                  num_workders=16, 
                                  shuffle=False):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')    

    device = torch.device(
        "cuda:"+str(choose_gpu) if torch.cuda.is_available() else "cpu"
    )

    if encoder == 'CLIP' and choose_encoder_version == "ViT-L/14":
        model, _, vis_processors = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        txt_processors = open_clip.get_tokenizer('ViT-L-14')
        
    elif encoder == 'CLIP' and choose_encoder_version == "ViT-B/32":
        model, _, vis_processors = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        txt_processors = open_clip.get_tokenizer('ViT-B-32')    
    else:
        raise("Choose one of the available encoders")


    print("Model loaded")
    model.to(device)
    model.eval()

    if "VERITE" in data_name_X:
        verite_df = pd.read_csv(data_path + '/VERITE_with_evidence.csv', index_col=0)
        verite_df = str_to_list(verite_df, list_columns=['captions', 'images_paths'])
        verite_df['match_index'] = verite_df.index.astype(str).tolist()        
        all_X_captions = idx_captions(verite_df, 'VERITE')
        all_X_images = idx_images(verite_df, 'VERITE')

        missing_images = all_X_images[all_X_images.X_image_path =='']
        all_X_images = all_X_images[~all_X_images.X_item_index.isin(missing_images.X_item_index)]
        
    else:
        print("Load data")
        vn_data = load_visual_news(data_path)

        train_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_train.csv', index_col=0)
        valid_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_valid.csv', index_col=0)
        test_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_test.csv', index_col=0)

        train_data = train_data.merge(vn_data[['id', 'caption']])
        valid_data = valid_data.merge(vn_data[['id', 'caption']])
        test_data = test_data.merge(vn_data[['id', 'caption']])

        train_data = train_data.merge(vn_data[['image_id', 'image_path']])
        valid_data = valid_data.merge(vn_data[['image_id', 'image_path']])
        test_data = test_data.merge(vn_data[['image_id', 'image_path']])

        del vn_data

        train_data = str_to_list(train_data, list_columns=['captions', 'images_paths'])
        valid_data = str_to_list(valid_data, list_columns=['captions', 'images_paths'])
        test_data = str_to_list(test_data, list_columns=['captions', 'images_paths'])

        train_data.match_index = 'train_' + train_data.match_index.astype(str)
        valid_data.match_index = 'valid_' + valid_data.match_index.astype(str)
        test_data.match_index = 'test_' + test_data.match_index.astype(str)

        train_X_captions = idx_captions(train_data, 'train')
        valid_X_captions = idx_captions(valid_data, 'valid')
        test_X_captions = idx_captions(test_data, 'test')

        train_X_images = idx_images(train_data, 'train')
        valid_X_images = idx_images(valid_data, 'valid')
        test_X_images = idx_images(test_data, 'test')

        all_X_captions = pd.concat([train_X_captions, valid_X_captions, test_X_captions])
        all_X_images = pd.concat([train_X_images, valid_X_images, test_X_images])
        missing_images = all_X_images[all_X_images.X_image_path =='']
        all_X_images = all_X_images[~all_X_images.X_item_index.isin(missing_images.X_item_index)]
    
    img_dataloader, txt_dataloader = prepare_source_dataloaders(all_X_images, 
                                        all_X_captions, 
                                        vis_processors, 
                                        txt_processors,  
                                        encoder_version,
                                        batch_size, 
                                        num_workders, 
                                        shuffle)

    text_ids, image_ids, all_text_features, all_visual_features = [], [], [], []
    
    if not os.path.isdir(output_path + 'temp_visual_features'):
        os.makedirs(output_path + 'temp_visual_features')

    print("Extract features from image evidence. Save the batch as a numpy file")
    batch_count = 0
    with torch.no_grad():

        for idx, img in tqdm(img_dataloader):

            image_features = model.encode_image(img.to(device))
            image_features = image_features.reshape(image_features.shape[0], -1).cpu().detach().numpy()
            
            np.save(output_path + 'temp_visual_features/' + data_name_X + '_' + encoder.lower() + '_' + str(batch_count), image_features) 

            batch_count += 1
            image_ids.extend(idx)

            del image_features
            del img

    print("Save: ", output_path)
    image_ids = np.stack(image_ids)
    np.save(output_path + data_name_X + "_image_ids_" + encoder_version +".npy", image_ids)    
    
    print("Load visual features (numpy files) and concatenate into a single file")
    image_embeddings = []
    for batch_count in range(img_dataloader.__len__()):

        print(batch_count, end='\r')
        image_features = np.load(output_path + 'temp_visual_features/' + data_name_X + '_' + encoder.lower() + '_' + str(batch_count) + '.npy') 
        image_embeddings.extend(image_features)
    
    image_embeddings = np.array(image_embeddings)
    image_ids = np.load(output_path + data_name_X + "_image_ids_" + encoder_version +".npy")
    np.save(output_path + data_name_X + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy", image_embeddings) 
    
    print("Extract features from text evidence")
    with torch.no_grad():
        for idx, txt in tqdm(txt_dataloader):

            text_features = model.encode_text(txt.squeeze(1).to(device))            
            text_features = text_features.reshape(text_features.shape[0], -1)
            all_text_features.extend(text_features.cpu().detach().numpy())
            text_ids.extend(idx)
        
    all_text_features = np.stack(all_text_features)
    
    np.save(output_path + data_name_X + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy", all_text_features)    
    text_ids = np.stack(text_ids)
    np.save(output_path + data_name_X + "_text_ids_" + encoder_version +".npy", text_ids)   


def calc_sim(q_emb, X_items):
    
    cos_sim = cosine_similarity(q_emb.reshape(1, -1), X_items) # Calculate the cosine similarity
    cos_sim = pd.Series(cos_sim[0], index=X_items.index) # Convert the cosine similarities to a pandas Series
    cos_sim = cos_sim.sort_values(ascending=False) # Sort the cosine similarities in descending order   

    return cos_sim


def rank_X_items(input_df, image_cols, text_cols, X_image_embeddings, X_text_embeddings, image_embeddings, text_embeddings):

    img_most_similar_items = []
    txt_most_similar_items = []
    
    for (row) in tqdm(input_df.itertuples(), total=input_df.shape[0]):
              
        match_index_img = [string for string in image_cols if string.startswith(row.match_index + '_')]
        match_index_txt = [string for string in text_cols if string.startswith(row.match_index + '_')]
        
        if match_index_img != []:
            img_items = X_image_embeddings[match_index_img].T
            q_img_emb = image_embeddings[row.image_id].values
            img_cos_sim = calc_sim(q_img_emb, img_items)
            img_most_similar_items.append(
                {
                    'img_ranked_items': img_cos_sim.index.tolist(), 
                    'img_sim_scores': img_cos_sim.values.tolist()
                }
            )
            
        else:
            img_most_similar_items.append(
                {
                    'img_ranked_items': [], 
                    'img_sim_scores': []
                }
            )
                        
        if match_index_txt != []:
                       
            txt_items = X_text_embeddings[match_index_txt].T
            q_txt_emb = text_embeddings[row.image_id].values
            txt_cos_sim = calc_sim(q_txt_emb, txt_items)
            txt_most_similar_items.append(
                {
                    'txt_ranked_items': txt_cos_sim.index.tolist(), 
                    'txt_sim_scores': txt_cos_sim.values.tolist()
                }
            )
        else:
            txt_most_similar_items.append(
                {
                    'txt_ranked_items': [], 
                    'txt_sim_scores': []
                }
            )           
            
    return pd.DataFrame(img_most_similar_items), pd.DataFrame(txt_most_similar_items)


def rank_evidence(data_path, data_name, data_name_X, output_path, encoder, choose_encoder_version):
    print("Load data")
    vn_data = load_visual_news(data_path)

    train_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_train.csv', index_col=0)
    valid_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_valid.csv', index_col=0)
    test_data = pd.read_csv(data_path + 'news_clippings/merged_balanced_test.csv', index_col=0)

    train_data = train_data.merge(vn_data[['id', 'caption']])
    valid_data = valid_data.merge(vn_data[['id', 'caption']])
    test_data = test_data.merge(vn_data[['id', 'caption']])   

    train_data = train_data.merge(vn_data[['image_id', 'image_path']])
    valid_data = valid_data.merge(vn_data[['image_id', 'image_path']])
    test_data = test_data.merge(vn_data[['image_id', 'image_path']])

    train_data[['id', 'image_id', 'match_index']] = train_data[['id', 'image_id', 'match_index']].astype('str')
    valid_data[['id', 'image_id', 'match_index']] = valid_data[['id', 'image_id', 'match_index']].astype('str')
    test_data[['id', 'image_id', 'match_index']] = test_data[['id', 'image_id', 'match_index']].astype('str')    
    
    train_data.match_index = "train_" + train_data.match_index
    valid_data.match_index = "valid_" + valid_data.match_index    
    test_data.match_index = "test_" + test_data.match_index  

    train_data = train_data[train_data.id != '111288']
    train_data = train_data[train_data.image_id != '111288']        
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')

    X_image_embeddings = np.load(output_path + data_name_X + '_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_image_ids = np.load(output_path + data_name_X + "_image_ids_" + encoder_version +".npy")
    X_image_embeddings = pd.DataFrame(X_image_embeddings, index=X_image_ids).T
    X_image_embeddings.columns = X_image_embeddings.columns.astype('str')

    X_text_embeddings = np.load(output_path + data_name_X + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_text_ids = np.load(output_path + data_name_X + "_text_ids_" + encoder_version +".npy")

    X_text_embeddings = pd.DataFrame(X_text_embeddings, index=X_text_ids).T
    X_text_embeddings.columns = X_text_embeddings.columns.astype('str')  
    X_text_embeddings = X_text_embeddings.loc[:, ~X_text_embeddings.columns.duplicated()]

    all_ids = np.concatenate([train_data.id, valid_data.id, test_data.id])
    all_image_ids = np.concatenate([train_data.image_id, valid_data.image_id, test_data.image_id])
    keep_ids = np.unique(np.concatenate([all_ids, all_image_ids]))

    image_embeddings, text_embeddings = load_features(data_path, data_name, encoder, encoder_version.lower(), keep_ids)
    
    image_cols = X_image_embeddings.columns.tolist()
    text_cols = X_text_embeddings.columns.tolist()
    
    valid_ranked_X_img, valid_ranked_X_txt = rank_X_items(valid_data, 
                                                          image_cols, 
                                                          text_cols,
                                                          X_image_embeddings, 
                                                          X_text_embeddings,
                                                          image_embeddings,
                                                          text_embeddings
                                                         )
    valid_data = pd.concat([valid_data, valid_ranked_X_img], axis=1)
    valid_data = pd.concat([valid_data, valid_ranked_X_txt], axis=1)

    valid_data.to_csv(data_path + 'news_clippings/merged_balanced_valid_ranked_' + encoder.lower() + "_" + encoder_version + '.csv')
    
    test_ranked_X_img, test_ranked_X_txt = rank_X_items(test_data, 
                                                        image_cols, 
                                                        text_cols, 
                                                        X_image_embeddings, 
                                                        X_text_embeddings,
                                                        image_embeddings,
                                                        text_embeddings                                                       
                                                       )
    test_data = pd.concat([test_data, test_ranked_X_img], axis=1)
    test_data = pd.concat([test_data, test_ranked_X_txt], axis=1)
    test_data.to_csv(data_path + 'news_clippings/merged_balanced_test_ranked_' + encoder.lower() + "_" + encoder_version + '.csv')
    
    train_ranked_X_img, train_ranked_X_txt = rank_X_items(train_data, 
                                                          image_cols, 
                                                          text_cols, 
                                                          X_image_embeddings, 
                                                          X_text_embeddings,
                                                          image_embeddings,
                                                          text_embeddings
                                                         )
    train_data = pd.concat([train_data, train_ranked_X_img], axis=1)
    train_data = pd.concat([train_data, train_ranked_X_txt], axis=1)
    train_data.to_csv(data_path + 'news_clippings/merged_balanced_train_ranked_' + encoder.lower() + "_" + encoder_version + '.csv')
   

def re_rank_verite(data_path, data_name, output_path, encoder='CLIP', choose_encoder_version='ViT-B/32'):
    
    data = pd.read_csv(data_path + 'VERITE_with_evidence.csv', index_col=0)
    data['match_index'] = data.index.astype(str).tolist()            
    data["id"] = data.index.astype(str).tolist()
    data["image_id"] = data.index.astype(str).tolist()   
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')
    
    verite_text_embeddings = np.load(data_path + "VERITE_" + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy").astype('float32')
    verite_image_embeddings = np.load(data_path + "VERITE_" + encoder.lower() +"_image_embeddings_" + encoder_version + ".npy").astype('float32')

    verite_image_embeddings = pd.DataFrame(verite_image_embeddings, index=[str(x) for x in range(1001)]).T
    verite_text_embeddings = pd.DataFrame(verite_text_embeddings, index=[str(x) for x in range(1001)]).T
    
    X_image_embeddings = np.load(output_path + data_name + '_external_' + encoder.lower() + "_image_embeddings_" + encoder_version + ".npy")
    X_image_ids = np.load(output_path + data_name + "_external_image_ids_" + encoder_version +".npy")
    X_image_embeddings = pd.DataFrame(X_image_embeddings, index=X_image_ids).T
    X_image_embeddings.columns = X_image_embeddings.columns.astype('str')

    X_text_embeddings = np.load(output_path + data_name + '_external_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy")
    X_text_ids = np.load(output_path + data_name + "_external_text_ids_" + encoder_version +".npy")
    X_text_embeddings = pd.DataFrame(X_text_embeddings, index=X_text_ids).T
    X_text_embeddings.columns = X_text_embeddings.columns.astype('str')
    
    data = str_to_list(data, list_columns=['captions', 'images_paths'])
    
    image_cols = X_image_embeddings.columns.tolist()
    text_cols = X_text_embeddings.columns.tolist()

    ranked_X_img, ranked_X_txt = rank_X_items(data, 
                                              image_cols, 
                                              text_cols, 
                                              X_image_embeddings, 
                                              X_text_embeddings, 
                                              verite_image_embeddings, 
                                              verite_text_embeddings)

    ranked_data = pd.concat([data, ranked_X_img], axis=1)
    ranked_data = pd.concat([ranked_data, ranked_X_txt], axis=1)  

    ranked_data.to_csv(data_path + "VERITE_ranked_evidence_" + encoder.lower() + "_" + encoder_version +  ".csv")