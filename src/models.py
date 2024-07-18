import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob=0.1, final_output=False):
        super(mlp, self).__init__()
        
        self.layers_list = nn.ModuleList()
        
        for _ in range(num_layers):

            layer = nn.Sequential(
                nn.Linear(input_size, hidden_size),               
                nn.GELU(),                 
            )
            self.layers_list.append(layer)
            
            input_size = hidden_size
            
        if final_output:
            self.layers_list.append(nn.Linear(input_size, final_output)) 
                           
    def forward(self, x):
        
        for layer in self.layers_list:
            x = layer(x)
            
        return x 
    
    
class muse_mlp(nn.Module):
    def __init__(self, hidden_size, sims_to_keep, num_layers=1, dropout_prob=0.0):
        super(muse_mlp, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sims_to_keep = sims_to_keep        
        self.input_size = 1 + len(self.sims_to_keep)-1
        
        self.sim_mlp = mlp(self.input_size, self.hidden_size, self.num_layers)
                                   
    def forward(self, img, txt, img_X=None, txt_X=None):
    
        x = None
        
        if 'img_txt' in self.sims_to_keep:        
            img_txt = F.cosine_similarity(img, txt, dim=1)
            x = img_txt.unsqueeze(1)

        if 'img_X_img' in self.sims_to_keep:
            img_img_X = F.cosine_similarity(img, img_X.squeeze(1), dim=1)
        
            if x is None:
                x = img_img_X.unsqueeze(1)
            else:
                x = torch.cat([x, img_img_X.unsqueeze(1)], dim=1)

        if 'txt_X_img' in self.sims_to_keep:
            txt_X_img = F.cosine_similarity(txt, img_X.squeeze(1), dim=1)

            if x is None:    
                x = txt_X_img.unsqueeze(1)
            else:
                x = torch.cat([x, txt_X_img.unsqueeze(1)], dim=1)

        if 'img_X_txt' in self.sims_to_keep:
            img_X_txt = F.cosine_similarity(img, txt_X.squeeze(1), dim=1)
            if x is None:   
                x = img_X_txt.unsqueeze(1)
            else:
                x = torch.cat([x, img_X_txt.unsqueeze(1)], dim=1)

        if 'txt_X_txt' in self.sims_to_keep:
            txt_X_txt = F.cosine_similarity(txt, txt_X.squeeze(1), dim=1)
            
            if x is None:
                x = txt_X_txt.unsqueeze(1)
            else:
                x = torch.cat([x, txt_X_txt.unsqueeze(1)], dim=1)

        if 'X_img_X_txt' in self.sims_to_keep:
            X_img_X_txt = F.cosine_similarity(img_X.squeeze(1), txt_X.squeeze(1), dim=1)
            
            if x is None:            
                x = X_img_X_txt.unsqueeze(1)
            else:
                x = torch.cat([x, X_img_X_txt.unsqueeze(1)], dim=1)

        x = self.sim_mlp(x)

        return x     
        
class MUSE_MLP_CLF(nn.Module):
    def __init__(self, emb_dim, sims_to_keep, dropout=0.1, activation="gelu"):
        
        super(MUSE_MLP_CLF, self).__init__()
                
        self.emb_dim = emb_dim
        self.sims_to_keep = sims_to_keep        
        self.muse_mlp_component = muse_mlp(self.emb_dim, self.sims_to_keep)
        
        self.clf = mlp(self.muse_mlp_component.hidden_size, emb_dim, 1, dropout, final_output=1) 

    def forward(self, img, txt, img_X=None, txt_X=None):
        
        x = self.muse_mlp_component(img, txt, img_X, txt_X)  
        y = self.clf(x).flatten()
        
        return y

def combine_features(a, b, fusion_method):
    
    if "concat_1" in fusion_method:  
        x = torch.cat([a, b], dim=1)

    if 'add' in fusion_method:
        added = torch.add(a, b)
        x = torch.cat([x, added], axis=1)        

    if 'mul' in fusion_method:
        mult = torch.mul(a, b)
        x = torch.cat([x, mult], axis=1)        

    if 'sub' in fusion_method:
        sub = torch.sub(a, b)
        x = torch.cat([x, sub], axis=1)   
        
    return x   
    
class AITR(nn.Module):
    def __init__(self, emb_dim, fusion_method, use_evidence, use_muse, sims_to_keep, 
                 transformer_version, tf_h_l, tf_dim, pooling_method, 
                 dropout=0.1, activation="gelu"):
        super(AITR, self).__init__()
        
        clf_input_size = 0
        
        # General parameters
        self.emb_dim = emb_dim
        self.fusion_method = fusion_method
        self.use_evidence = use_evidence 
        self.transformer_version = transformer_version
        self.pooling_method = pooling_method        
                
        # Multimodal Similarity
        self.use_muse = use_muse
        self.sims_to_keep = sims_to_keep        
        
        if self.use_muse:
            self.muse_component = muse_mlp(self.emb_dim, self.sims_to_keep)                
            clf_input_size += self.muse_component.hidden_size
                    
        if self.transformer_version == "default":
            
            self.tf_head = tf_h_l[0]
            self.tf_layers = len(tf_h_l)
            self.tf_dim = tf_dim
            
            self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
            self.cls_token.requires_grad = True            
            
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.tf_head,
                    dim_feedforward=self.tf_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,                
                ),
                num_layers=self.tf_layers,
            )
            
            clf_input_size = emb_dim

        elif self.transformer_version == "aitr":    
            
            self.tf_dim = tf_dim
            transformer_list = nn.ModuleList()

            self.cls_token = nn.Parameter(torch.randn(self.emb_dim))
            self.cls_token.requires_grad = True                 
            
            for num_heads in tf_h_l:
                transformer_layer = nn.TransformerEncoderLayer(
                    d_model=emb_dim,
                    nhead=num_heads,
                    dim_feedforward=self.tf_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )
                transformer_list.append(transformer_layer)     

            self.transformer = transformer_list

            if self.pooling_method == "weighted_pooling":                
                self.wp_linear = nn.Linear(emb_dim, emb_dim)
                self.softmax = nn.Softmax(dim=-1)

            elif self.pooling_method == "attention_pooling":
                
                self.q_layer = nn.Linear(emb_dim, emb_dim)
                self.k_layer = nn.Linear(emb_dim, emb_dim)
                self.v_layer = nn.Linear(emb_dim, emb_dim)
                self.softmax = nn.Softmax(dim=-1)  
                
            clf_input_size = emb_dim

        self.clf_input_size = clf_input_size
        self.clf = mlp(self.clf_input_size, emb_dim, 1, dropout, final_output=1) 
    
    def forward(self, img, txt, img_X=None, txt_X=None):
        
        b_size = img.shape[0]
                    
        if self.use_muse:
            similarity_encoded = self.muse_component(img, txt, img_X, txt_X)  
                                                      
        cls_token = self.cls_token.expand(b_size, 1, -1) 
        
        fused_features = combine_features(img, txt, self.fusion_method) 
        fused_features = fused_features.reshape(b_size, -1, self.emb_dim)

        if self.use_muse:
            x = torch.cat([cls_token, fused_features, similarity_encoded.unsqueeze(1)], dim=1)
        else:
            x = torch.cat([cls_token, fused_features], dim=1)                
    
        if self.use_evidence:                        
            x = torch.cat([x, img_X, txt_X], dim=1)  

        if self.transformer_version == "default":
            x = self.transformer(x)[:,0,:]

        elif self.transformer_version == "aitr":

            outputs = []
            for layer in self.transformer:                                    
                x = layer(x)
                outputs.append(x[:,0,:])
            
            x_t = torch.stack(outputs, dim=1)  
            
            if self.pooling_method == "max_pooling":                  
                x, _ = torch.max(x_t, dim=1)       

            elif self.pooling_method == "weighted_pooling":
                
                w = self.softmax(self.wp_linear(x_t))
                x = torch.sum(w * x_t, dim=1)         

            elif self.pooling_method == "attention_pooling":                   
                
                Q = self.q_layer(x_t)  
                K = self.k_layer(x_t)     
                V = self.v_layer(x_t)
                
                attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5) 
                attention_weights = self.softmax(attention_scores)
                attention_output = torch.matmul(attention_weights, V) 

                x = attention_output.mean(1)

        else:
            raise Exception("Choose between Transformer: default or aitr")
        
        y = self.clf(x).flatten()
        
        return y