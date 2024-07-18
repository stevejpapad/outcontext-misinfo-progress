import os 
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from utils import load_ranked_evidence, load_evidence_features, load_ranked_verite, load_features, calculate_muse

def prepare_muse_data(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version):

    out_path = evidence_path + data_name + "_sims_train" + "_" + encoder.lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    
    train_data, valid_data, test_data = load_ranked_evidence(encoder, encoder_version, data_path, data_name, data_name_X)
    X_image_embeddings, X_text_embeddings = load_evidence_features(encoder, encoder_version, evidence_path, data_name, data_name_X)
    verite_test, verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings = load_ranked_verite(encoder, encoder_version, verite_path)
    image_embeddings, text_embeddings = load_features(data_path=data_path, data_name="_news_clippings", encoder=encoder, encoder_version=encoder_version)
    
    for input_data, split_name in [(train_data, "train"), (valid_data, "valid"), (test_data, "test")]:
        out_path = evidence_path + data_name + "_sims_" + split_name + "_" + encoder.replace("-", "").replace("/", "").lower() + "_" + ENCODER_VERSION.replace("-", "").replace("/", "").lower()
        calculate_muse(input_data, split_name, image_embeddings, text_embeddings,  X_image_embeddings, X_text_embeddings, out_path)
    
    out_path = verite_path + "sims_" + encoder.lower() + "_" + ENCODER_VERSION.replace("-", "").replace("/", "").lower()
    calculate_muse(verite_test, "verite", verite_image_embeddings, verite_text_embeddings, X_verite_image_embeddings, X_verite_text_embeddings, out_path)

def load_muse_data(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version):

    out_path = evidence_path + data_name + "_sims_train" + "_" + encoder.lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    
    if not os.path.isfile(out_path + ".csv"):
        prepare_muse_data(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version)     
    
    out_path = evidence_path + data_name + "_sims_train" + "_" + encoder.lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    train_data = pd.read_csv(out_path + ".csv", index_col=0)
    train_data.falsified = train_data.falsified.astype('int')
    
    out_path = evidence_path + data_name + "_sims_valid" + "_" + encoder.lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    valid_data = pd.read_csv(out_path + ".csv", index_col=0)
    valid_data.falsified = valid_data.falsified.astype('int')
    
    out_path = evidence_path + data_name + "_sims_test" + "_" + encoder.lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    test_data = pd.read_csv(out_path + ".csv", index_col=0)
    test_data.falsified = test_data.falsified.astype('int')
    
    out_path = verite_path + "sims_" + "verite" + "_" + encoder.lower() + "_" + encoder_version.replace("-", "").replace("/", "").lower()
    verite_data = pd.read_csv(out_path + ".csv", index_col=0)
    verite_data = verite_data[verite_data.falsified.isin([0,2])]
    verite_data.falsified = verite_data.falsified.replace(2, 1)

    return train_data, valid_data, test_data, verite_data

def train_eval_MUSE(model_name, X_train, y_train, X_test, y_test, X_verite, y_verite):
   
    if model_name == "DT":
        model = DecisionTreeClassifier(random_state=42, max_depth=7)

    elif model_name == "RF":
        model = RandomForestClassifier(random_state=42)
        
    else:
        raise Exception("Choose one of the available options: DT or RF")
    
    model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
            
    pred = model.predict(X_test)
    test_accuracy = metrics.accuracy_score(pred, y_test)
    
    verite_pred = model.predict(X_verite)
    verite_accuracy = metrics.accuracy_score(verite_pred, y_verite)
    
    return test_accuracy, verite_accuracy, feature_importance

def print_results(test_acc, verite_acc, feature_names, feature_importance):

    print("NewsCLIPpings accuracy: ", round(test_acc * 100, 2))    
    print("VERITE-OOC accuracy: ", round(verite_acc * 100, 2))
    
    print("\nSimilarity Importance")
    for feature, importance in zip(feature_names, feature_importance):
        print(f"{feature}: {importance:.4f}")

def muse_similarity_importance(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version):

    encoder_version = encoder_version.replace("-", "").replace("/", "")        

    train_data, valid_data, test_data, verite_data = load_muse_data(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version)
    
    y_train = train_data.falsified
    feature_names = ['img_txt','img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'] 
    
    X_train = train_data[feature_names]
    y_test = test_data.falsified
    X_test = test_data[feature_names]
    
    y_verite = verite_data.falsified
    X_verite = verite_data[feature_names]

    test_acc, verite_acc, feature_importance = train_eval_MUSE("DT", X_train, y_train, X_test, y_test, X_verite, y_verite)
    print("Decition Tree")
    print_results(test_acc, verite_acc, feature_names, feature_importance)
    
    test_acc, verite_acc, feature_importance = train_eval_MUSE("RF", X_train, y_train, X_test, y_test, X_verite, y_verite)
    print("Random Forest")
    print_results(test_acc, verite_acc, feature_names, feature_importance)


def limited_data(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version):

    encoder_version = encoder_version.replace("-", "").replace("/", "")    

    train_data, valid_data, test_data, verite_data = load_muse_data(data_path, evidence_path, verite_path, data_name, data_name_X, encoder, encoder_version)    
    
    results = []
    
    for frac in [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    
        X = train_data.sample(frac=frac, random_state=0)
        y_train = X.falsified
    
        feature_names = ['img_txt','img_X_img', 'txt_X_img', 'img_X_txt', 'txt_X_txt', 'X_img_X_txt'] 
        
        X_train = X[feature_names]   
        y_test = test_data.falsified
        X_test = test_data[feature_names]
        y_verite = verite_data.falsified
        X_verite = verite_data[feature_names]
    
        test_acc, verite_acc, _ = train_eval_MUSE("RF", X_train, y_train, X_test, y_test, X_verite, y_verite)
    
        if frac >= 0.01:
            frac = int(frac * 100)
        else:
            frac = frac * 100
        
        results.append([str(frac)+"%", round(test_acc*100, 2), round(verite_acc*100,2)])
        

    # FIGURE for MUSE-RF
    df = pd.DataFrame(results, columns=["Percentage", "NewsCLIPpings", "VERITE"])
    
    df['Percentage'] = pd.Categorical(df['Percentage'], 
                                      categories=['0.01%', '0.05%', '0.1%', '0.5%', '1%', '5%', '10%', '25%', '50%', '75%', '100%'], 
                                      ordered=True)
    
    df_melted = df.melt(id_vars='Percentage', var_name='dataset', value_name='performance')
    
    plt.figure(figsize=(8, 2))
    ax = sns.lineplot(x='Percentage', y='performance', hue='dataset', style='dataset', markers=['o', 'X'], data=df_melted, palette='viridis')
    ax.legend(loc='lower right', framealpha=1)
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.annotate(f'{y:.2f}', (x, y),
                        ha='center', 
                        va='bottom', 
                        fontsize=10, 
                        color='black', 
                        xytext=(0, 5),
                        textcoords='offset points')
    plt.ylabel('Accuracy')
    plt.ylim(50, 100)
    
    plt.savefig("docs/muse_rf_lim.png")   
    