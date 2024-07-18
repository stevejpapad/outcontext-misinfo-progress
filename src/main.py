from aitr_experiment import run_aitr
from feature_extraction import extract_encoder_features
from evidence_preparation import extract_features_for_evidence, rank_evidence, re_rank_verite
from muse_mlp_experiment import run_muse_mlp
from muse_ml_experiment import muse_similarity_importance, limited_data

ENCODER='CLIP'
ENCODER_VERSION = "ViT-L/14"
DATA_PATH = '' 
EVIDENCE_PATH = 'news_clippings/'
DATA_NAME = 'news_clippings_balanced'
DATA_NAME_X = "news_clippings_balanced_external_info"
images_path=DATA_PATH+'VisualNews/origin/'
VERITE_PATH = 'VERITE/'

# Extract CLIP features for NewsCLIPpings
extract_encoder_features(data_path=DATA_PATH, images_path=DATA_PATH+'VisualNews/origin/', data_name=DATA_NAME, output_path=EVIDENCE_PATH)

# Extract CLIP features for VERITE
extract_encoder_features(data_path=VERITE_PATH, images_path=VERITE_PATH, data_name="VERITE", output_path=VERITE_PATH)

# Extract CLIP features for NewsCLIPpings Evidence
extract_features_for_evidence(data_path=DATA_PATH, output_path=EVIDENCE_PATH, data_name_X=DATA_NAME_X, encoder=ENCODER, choose_encoder_version=ENCODER_VERSION, choose_gpu=0)

# Re-rank NewsCLIPpings evidence
rank_evidence(data_path=DATA_PATH, data_name=DATA_NAME, data_name_X=DATA_NAME_X, output_path=EVIDENCE_PATH, encoder=ENCODER, choose_encoder_version=ENCODER_VERSION)

# Extract CLIP features for VERITE Evidence
extract_features_for_evidence(data_path=VERITE_PATH, output_path=VERITE_PATH, data_name_X="VERITE_external", encoder=ENCODER, choose_encoder_version=ENCODER_VERSION, choose_gpu=0)

# Re-rank VERITE evidence
re_rank_verite(data_path=VERITE_PATH, data_name="VERITE", output_path=VERITE_PATH, encoder=ENCODER, choose_encoder_version=ENCODER_VERSION)

# Experiment for Table 3: Feature/Similarity importance and performance by MUSE-RF and MUSE-DT
muse_similarity_importance(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, DATA_NAME, DATA_NAME_X, ENCODER, ENCODER_VERSION)

# Experiment for Figure 3: Performance of MUSE-RF with limited training data
limited_data(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, DATA_NAME, DATA_NAME_X, ENCODER, ENCODER_VERSION)

# Experiments for Table 4 (Ablation of MUSE-MLP)
run_muse_mlp(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, full_ablation=True, encoder = ENCODER, encoder_version = ENCODER_VERSION)

# Experiments for Table 1, 2, 5 [AITR ablation]

# AITR with attention pooling and MUSE 
run_aitr(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0, 
         use_muse = True, use_evidence = 1, transformer_version = "aitr", pooling_method = "attention_pooling")

# AITR with attention pooling without MUSE 
run_aitr(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0, 
         use_muse = False, use_evidence = 1, transformer_version = "aitr", pooling_method = "attention_pooling")

# AITR with weighted pooling and MUSE 
run_aitr(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0, 
         use_muse = True, use_evidence = 1, transformer_version = "aitr", pooling_method = "weighted_pooling")

# AITR with max pooling and MUSE 
run_aitr(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0, 
         use_muse = True, use_evidence = 1, transformer_version = "aitr", pooling_method = "max_pooling")

# Default transformer encoder with MUSE 
run_aitr(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0, 
         use_muse = True, use_evidence = 1, transformer_version = "default", pooling_method = None)

# Default transformer encoder without MUSE 
run_aitr(DATA_PATH, EVIDENCE_PATH, VERITE_PATH, encoder = 'CLIP', encoder_version = 'ViT-L/14', choose_gpu = 0, 
         use_muse = False, use_evidence = 1, transformer_version = "default", pooling_method = None)