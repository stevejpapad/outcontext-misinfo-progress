import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from string import digits
import open_clip
from torch.utils.data import DataLoader


class DatasetIteratorSource(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data,
        images_path,
        vis_processors,
        txt_processors,
        encoder_version
    ):
        self.input_data = input_data
        self.images_path = images_path
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors
        self.encoder_version = encoder_version

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        current = self.input_data.iloc[idx]

        img_path = self.images_path + current.image_path.split('./')[-1]
        image = Image.open(img_path).convert('RGB')

        max_size = 400
        width, height = image.size

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Resize the image
        image = image.resize((new_width, new_height))
        
        img = self.vis_processors(image)
        txt = self.txt_processors(current.caption)
                    
        idx = current.id
        
        return idx, img, txt
    
def prepare_source_dataloader(input_data, vis_processors, txt_processors, encoder_version, images_path, batch_size, num_workers, shuffle):
    
    dg = DatasetIteratorSource(input_data, images_path, vis_processors, txt_processors, encoder_version)

    dataloader = DataLoader(
        dg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader   

def load_dataset(data_path, data_name):
        
        
    print("Data path: ", data_path)
    print("Load: ", data_name)
        
    if 'news_clippings_balanced' == data_name:
                
        train_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/train.json"))
        valid_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/val.json"))
        test_data = json.load(open(data_path + "news_clippings/data/news_clippings/data/merged_balanced/test.json"))

        train_data = pd.DataFrame(train_data["annotations"])
        valid_data = pd.DataFrame(valid_data["annotations"])
        test_data = pd.DataFrame(test_data["annotations"])

        nc_data = pd.concat([train_data, valid_data, test_data])
        
        all_ids = np.concatenate([nc_data.id.unique(), nc_data.image_id.unique()])
        keep_ids = list(np.unique(all_ids))

        vn_data = json.load(open(data_path + 'VisualNews/origin/data.json'))
        vn_data = pd.DataFrame(vn_data)

        data = vn_data[vn_data.id.isin(keep_ids)]
        
    elif 'VisualNews' in data_name: 
        data = json.load(open(data_path + 'data.json'))
        data = pd.DataFrame(data)
        
    elif 'VERITE' in data_name:
        data = pd.read_csv(data_path +'VERITE.csv', index_col=0)        
        data['id'] = [i for i in range(data.shape[0])]
    
    else:
        raise ValueError("data_name does not match any available dataset")
                
    return data

def extract_encoder_features(data_path, 
                             images_path, 
                             data_name, 
                             output_path, 
                             encoder='CLIP', 
                             choose_encoder_version="ViT-L/14", 
                             batch_size=100,
                             choose_gpu=0, 
                             num_workders=16, 
                             shuffle=False):
                   
    os.environ["CUDA_VISIBLE_DEVICES"] = str(choose_gpu)
    
    device = torch.device(
        "cuda:"+str(choose_gpu) if torch.cuda.is_available() else "cpu"
    )
      
    data = load_dataset(data_path, data_name)

    save_id = 'id' in data.columns
    
    if encoder == 'CLIP' and choose_encoder_version == "ViT-L/14":
        model, _, vis_processors = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        txt_processors = open_clip.get_tokenizer('ViT-L-14')
    
    elif encoder == 'CLIP' and choose_encoder_version == "ViT-B/32":
        model, _, vis_processors = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        txt_processors = open_clip.get_tokenizer('ViT-B-32')
    else:
        raise Exception("Choose one of the available encoders")

    data_loader = prepare_source_dataloader(data, 
                                            vis_processors, 
                                            txt_processors, 
                                            choose_encoder_version, 
                                            images_path, 
                                            batch_size, 
                                            num_workders, 
                                            shuffle)
        
    print("Model loaded")
    model.to(device)
    model.eval()
    
    if not os.path.isdir(output_path + 'temp_visual_features'):
        os.makedirs(output_path + 'temp_visual_features')
    
    if not os.path.isdir(output_path + 'temp_textual_features'):
        os.makedirs(output_path + 'temp_textual_features')
        
    all_ids, all_text_features, all_visual_features = [], [], []
    
    batch_count = 0
    with torch.no_grad():
    
        for idx, img, txt in tqdm(data_loader):
                            
            image_features = model.encode_image(img.to(device))
            text_features = model.encode_text(txt.squeeze(1).to(device))
            
            image_features = image_features.reshape(image_features.shape[0], -1)
            text_features = text_features.reshape(text_features.shape[0], -1)
            
            image_features = image_features.cpu().detach().numpy()
            np.save(output_path + 'temp_visual_features/' + data_name + '_' + encoder.lower() + '_' + str(batch_count), image_features) 
            
            text_features = text_features.cpu().detach().numpy()
            np.save(output_path + 'temp_textual_features/' + data_name + '_' + encoder.lower() + '_' + str(batch_count), text_features) 
            batch_count += 1
            
            all_ids.extend(idx)
            
            del image_features
            del text_features
            del img
    
            
    print("Save: ", output_path)
    
    encoder_version = choose_encoder_version.replace('-', '').replace('/', '')
    
    if save_id:
        all_ids = np.stack(all_ids)
        np.save(output_path + data_name + "_item_ids_" + encoder_version +".npy", all_ids)
        
    # LOAD extracted visual features and combine then into a single numpy file
    image_embeddings =[]
    for batch_count in range(data_loader.__len__()):
    
        print(batch_count, end='\r')
        image_features = np.load(output_path + 'temp_visual_features/' + data_name + '_' + encoder.lower() + '_' + str(batch_count) + '.npy') 
    
        image_embeddings.extend(image_features)
        
    image_embeddings = np.array(image_embeddings)
    np.save(output_path + data_name + '_' + encoder.lower() + "_image_embeddings_" + encoder_version  + ".npy", image_embeddings) 
        
    # LOAD extracted textual features and combine then into a single numpy file
    text_embeddings =[]
    for batch_count in range(data_loader.__len__()):
    
        print(batch_count, end='\r')
        text_features = np.load(output_path + 'temp_textual_features/' + data_name + '_' + encoder.lower() + '_' + str(batch_count) + '.npy') 
    
        text_embeddings.extend(text_features)
        
    text_embeddings = np.array(text_embeddings)
    np.save(output_path + data_name + '_' + encoder.lower() + "_text_embeddings_" + encoder_version + ".npy", text_embeddings) 

