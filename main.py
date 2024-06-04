################################
# This script provide a demo of MSI-DTI
from tensorflow.python.keras.optimizers import Adam, Adagrad, Adamax,SGD
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from datetime import datetime
from tensorflow import keras
from sklearn.decomposition import PCA
from lol import LOL
from tensorflow.python.keras.callbacks import EarlyStopping
from deepctr.models import NFM,AutoInt
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from cvae import cvae


os.environ["CUDA_VISIBLE_DEVICES"] = "9"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Choose the dataset (DTKG, Davis, KIBA, DrugBank)
dataset = "KIBA"

# load data
################################################################
# data example: yamanishi_08
dt_08 = pd.read_csv(f'./datasets/{dataset}/dti_all.txt',
                    delimiter='\t', header=None)
dt_08.columns = ['head', 'relation', 'tail']

network = pd.read_csv(f'./datasets/{dataset}/network.txt',
                  delimiter='\t', header=None)

network.index = range(len(network))
network.columns = ['head', 'relation', 'tail']

head_le = LabelEncoder()
tail_le = LabelEncoder()
head_le.fit(dt_08['head'].values)
tail_le.fit(dt_08['tail'].values)

mms = MinMaxScaler(feature_range=(0, 1))
ss = StandardScaler()

# descriptors preparation
fp_id = pd.read_csv(f'./datasets/{dataset}/drugs.csv')['drug_id']
df_proseq = pd.read_csv(f'./datasets/{dataset}/proteins.csv')
df_proseq.columns = ['pro_id', 'pro_ids', 'seq']
pro_id = df_proseq['pro_id']


#############drug features##########################
# drug_feats1 = np.loadtxt(f'./datasets/{dataset}/smiles/atom_pair.csv', delimiter=',')#1
drug_feats2 = np.loadtxt(f'./datasets/{dataset}/smiles/avalon.csv', delimiter=',') #512
#drug_feats3 = np.loadtxt(f'./datasets/{dataset}/smiles/ecfp2.csv', delimiter=',')#1024
drug_feats4 = np.loadtxt(f'./datasets/{dataset}/smiles/maccs.csv', delimiter=',')#167
#drug_feats5 = np.loadtxt(f'./datasets/{dataset}/smiles/molecule_descriptors.csv', delimiter=',')#208
drug_feats6 = np.loadtxt(f'./datasets/{dataset}/smiles/morgan.csv', delimiter=',')#1024
#drug_feats7 = np.loadtxt(f'./datasets/{dataset}/smiles/rdk.csv', delimiter=',')#2048
# drug_feats8 = np.loadtxt(f'./datasets/{dataset}/smiles/torsions.csv', delimiter=',')#1
drug_feats9 = np.loadtxt(f'./datasets/{dataset}/smiles/mol2vec.csv', delimiter=',')#300
#drug_feats10 = np.loadtxt(f'./datasets/{dataset}/smiles/rdkit.txt', delimiter='\t')#208
drug_feats11 = np.loadtxt(f'./datasets/{dataset}/smiles/mol_graph2vec.csv', delimiter=',')#128

# print("start pca drug feats")
# drug_feats2 = PCA(n_components=256).fit_transform(drug_feats2)
# drug_feats2 = mms.fit_transform(drug_feats2)
# drug_feats6 = PCA(n_components=256).fit_transform(drug_feats6)
# drug_feats6 = mms.fit_transform(drug_feats6)
#drug_feats10 = PCA(n_components=104).fit_transform(drug_feats10)
#drug_feats10 = mms.fit_transform(drug_feats10)
# print("Done pca drug feats")

#############protein features##########################
#pro_feats1 = np.loadtxt(f'./datasets/{dataset}/seq/new_cksaap.csv', delimiter=',')#2400
pro_feats2 = np.loadtxt(f'./datasets/{dataset}/seq/new_ctdc.csv', delimiter=',')#39
pro_feats3 = np.loadtxt(f'./datasets/{dataset}/seq/new_ctdt.csv', delimiter=',')#39
pro_feats4 = np.loadtxt(f'./datasets/{dataset}/seq/new_ctdd.csv', delimiter=',')#195
#pro_feats5 = np.loadtxt(f'./datasets/{dataset}/seq/new_ctriad.csv', delimiter=',')#343
pro_feats6 = np.loadtxt(f'./datasets/{dataset}/seq/new_geary.csv', delimiter=',')#72
pro_feats7 = np.loadtxt(f'./datasets/{dataset}/seq/new_paac.csv', delimiter=',')#28
pro_feats8 = np.loadtxt(f'./datasets/{dataset}/seq/new_qsorder.csv', delimiter=',')#44
pro_feats9 = np.loadtxt(f'./datasets/{dataset}/seq/cpcprot.csv', delimiter=',')


drug_feats = pd.concat([pd.DataFrame(drug_feats2),pd.DataFrame(drug_feats4),pd.DataFrame(drug_feats6),pd.DataFrame(drug_feats9),pd.DataFrame(drug_feats11)],axis=1)
pro_feats = pd.concat([pd.DataFrame(pro_feats2),pd.DataFrame(pro_feats3),pd.DataFrame(pro_feats4),pd.DataFrame(pro_feats6),pd.DataFrame(pro_feats7),pd.DataFrame(pro_feats8),pd.DataFrame(pro_feats9)],axis=1)

drug_feats = ss.fit_transform(drug_feats)
pro_feats_scaled = mms.fit_transform(pro_feats)
pro_feats_scaled2 = PCA(n_components=100).fit_transform(pro_feats_scaled)
pro_feats_scaled3 = mms.fit_transform(pro_feats_scaled2)

fp_df = pd.concat([fp_id, pd.DataFrame(drug_feats)], axis=1)


prodes_df = pd.concat([pro_id, pd.DataFrame(pro_feats)], axis=1)


# Function
################################################################

# If you want to test other dataset, just change the data path.

data_path = f'./datasets/{dataset}/data_folds/'

def load_data(i):
    train = pd.read_csv(data_path+'train_fold_'+str(i+1) +
                        '.csv')[['head', 'relation', 'tail', 'label']]
    train_pos = train[train['label'] == 1]
    test = pd.read_csv(data_path+'test_fold_'+str(i+1) +
                       '.csv')[['head', 'relation', 'tail', 'label']]


    data = pd.concat([train_pos,network])[['head', 'relation', 'tail']]
    print("load_data Completed")
    return train, train_pos, test, data


def roc_auc(y, pred):
    fpr, tpr, threshold = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, threshold, roc_auc


def pr_auc(y, pred):
    precision, recall, threshold = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return precision, recall, threshold, pr_auc

'''
def filter_unseen(test, train):
    train_entities = set(train['head']).union(set(train['tail']))
    test = test[test['head'].isin(train_entities) & test['tail'].isin(train_entities)]
    return test
'''

def feature_selection(data, threshold=0.9):

    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = data.drop(to_drop, axis=1)

    return selected_features

def get_scaled_embeddings(train_triples, test_triples, get_scaled, n_components):

    entity_df = pd.read_csv(f'./datasets/{dataset}/node_embeddings/CompGCN.csv', header=None)#CompGCN
    entity_emb_dict = {}
    for index, row in entity_df.iterrows():
        entity = row[0]
        entity_emb_dict[entity] = row.values[1:]

    entity_df1 = pd.read_csv(f'./datasets/{dataset}/node_embeddings/Attentionwalk.csv', header=None)#Attentionwalk
    #print(entity_df)
    entity_emb_dict1 = {}
    for index, row in entity_df1.iterrows():
        entity1 = row[0]
        entity_emb_dict1[entity1] = row.values[1:]

    # entity_df2 = pd.read_csv(f'./datasets/{dataset}/node_embeddings/Role2Vec.csv', header=None)#Role2Vec
    # #print(entity_df)
    # entity_emb_dict2 = {}
    # for index, row in entity_df2.iterrows():
    #     entity2 = row[0]
    #     entity_emb_dict2[entity2] = row.values[1:]


    [train_sub_embeddings, test_sub_embeddings] = [[entity_emb_dict[h] for h in x['head'].values] for x in [train_triples, test_triples]]
    [train_obj_embeddings, test_obj_embeddings] = [[entity_emb_dict[t] for t in x['tail'].values] for x in [train_triples, test_triples]]
    #print("train_sub_embeddings.shape")
    #print(train_sub_embeddings.shape)

    [train_sub_embeddings1, test_sub_embeddings1] = [[entity_emb_dict1[h] for h in x['head'].values] for x in [train_triples, test_triples]]
    [train_obj_embeddings1, test_obj_embeddings1] = [[entity_emb_dict1[t] for t in x['tail'].values] for x in [train_triples, test_triples]]


    train_sub_embeddings = np.concatenate([train_sub_embeddings,train_sub_embeddings1], axis=1)
    train_obj_embeddings = np.concatenate([train_obj_embeddings,train_obj_embeddings1], axis=1)

    test_sub_embeddings = np.concatenate([test_sub_embeddings,test_sub_embeddings1], axis=1)
    test_obj_embeddings = np.concatenate([test_obj_embeddings,test_obj_embeddings1], axis=1)

    train_feats = np.concatenate(
        [train_sub_embeddings, train_obj_embeddings], axis=1)
    test_feats = np.concatenate(
        [test_sub_embeddings, test_obj_embeddings], axis=1)
    
    # print("train_feats and test_feats isnan")
    # print(np.isnan(train_feats))
    # print(np.isnan(test_feats))

    
    # train_dense_features = ss.fit_transform(train_feats)
    # test_dense_features = ss.transform(test_feats)
    train_dense_features = feature_selection(train_feats)
    test_dense_features = feature_selection(test_feats)

    if get_scaled:
        pca = PCA(n_components=n_components)
        scaled_train_dense_features = pca.fit_transform(train_dense_features)
        scaled_pca_test_dense_features = pca.transform(test_dense_features)
    else:
        scaled_train_dense_features = train_dense_features
        scaled_pca_test_dense_features = test_dense_features
    print("get_embeddings Completed")
    return scaled_train_dense_features, scaled_pca_test_dense_features


def get_features(data, fp_df, prodes_df, use_pro):
    drug_features = pd.merge(data, fp_df, left_on='head', right_on='drug_id',how='left').iloc[:, 4:].values
    pro_features = pd.merge(data, prodes_df, how='left',left_on='tail', right_on='pro_id').iloc[:, 4:].values

    drug_features = feature_selection(drug_features)
    pro_features = feature_selection(pro_features)

    if use_pro:
        feature = np.concatenate([drug_features, pro_features], axis=1)
    else:
        feature = drug_features

    return feature



def get_attention_input(re_train_all, re_test_all, train_feats, test_feats, train_des, test_des, train_label_fs, test_label_fs, embedding_dim, pca_components):
    
    train_all_feats = np.concatenate([train_feats, train_des], axis=1)
    test_all_feats = np.concatenate([test_feats, test_des], axis=1)

    train_all_feats = np.nan_to_num(train_all_feats)
    test_all_feats = np.nan_to_num(test_all_feats)

    train_all_feats = ss.fit_transform(train_all_feats)
    test_all_feats = ss.transform(test_all_feats)
    
    
    # print("------------------------feature select-------------------------------")
    # print(type(train_label_fs))
    # print(type(test_label_fs))
    # lmao = LOL(n_components=512, svd_solver='full')
    # lmao.fit(train_all_feats, train_label_fs)

    # train_all_feats = lmao.fit_transform(train_all_feats, train_label_fs)
    # test_all_feats = lmao.transform(test_all_feats)
    
    train_all_feats_scaled = train_all_feats
    test_all_feats_scaled = test_all_feats

    feature_columns = [SparseFeat('head', re_train_all['head'].unique().shape[0], embedding_dim=embedding_dim,use_hash=True),
                       SparseFeat('tail', re_train_all['tail'].unique().shape[0], embedding_dim=embedding_dim,use_hash=True),
                       DenseFeat("feats", train_all_feats_scaled.shape[1]),
                       ]

    train_model_input = {'head': head_le.transform(re_train_all['head'].values),
                         'tail': tail_le.transform(re_train_all['tail'].values),
                         'feats': train_all_feats_scaled,
                         }

    test_model_input = {'head': head_le.transform(re_test_all['head'].values),
                        'tail': tail_le.transform(re_test_all['tail'].values),
                        'feats': test_all_feats_scaled,
                        }


    print("get_attention_input Completed")

    return feature_columns, train_model_input, test_model_input


def train_attention(feature_columns, train_model_input, train_label, test_model_input, y, patience):
    print("start train_attention")
    model = AutoInt(feature_columns, feature_columns, task='binary', att_layer_num=4, att_embedding_size=8, att_head_num=3,
                    att_res=True,dnn_hidden_units=(256, 256), dnn_activation='relu', l2_reg_linear=1e-4,
                    l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=1e-2, seed=2024,)
    model.compile(Adam(1e-4), "binary_crossentropy",
                     metrics=[keras.metrics.BinaryAccuracy(name='binary_accuracy'),keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')], )
    es = EarlyStopping(monitor='loss', patience=patience,
                       min_delta=0.0001, mode='min', restore_best_weights=False)
    


    history = model.fit(train_model_input, train_label,
                           batch_size=64, epochs=2000,
                           verbose=2,
                           callbacks=[es]
                           )
    


    pred_y = model.predict(test_model_input, batch_size=64)
    results = model.evaluate(test_model_input, y, batch_size=64)
    roc_fpr, roc_tpr, roc_threshold, roc_a = roc_auc(y, pred_y[:, 0])
    pr_precision, pr_recall, pr_threshold, pr_a = pr_auc(y, pred_y[:, 0])

    pred_y_binary = (pred_y > 0.5) * 1

    print("output_metrics")
    output_accuracy = metrics.accuracy_score(y, pred_y_binary[:, 0])
    output_precision = metrics.precision_score(y, pred_y_binary[:, 0])
    output_recall = metrics.recall_score(y, pred_y_binary[:, 0])
    output_f1 = metrics.f1_score(y, pred_y_binary[:, 0])

    print("output_accuracy",output_accuracy)
    print("output_precision",output_precision)
    print("output_recall",output_recall)
    print("output_f1",output_f1)    


    print("train_attention Completed")
    return roc_fpr, roc_tpr, roc_threshold, roc_a, pr_precision, pr_recall, pr_threshold, pr_a, pred_y[:, 0]


def train(i, embedding_dim, n_components, use_pro, patience,optimizer_params):
    print("start train")

    train, train_pos, test, data = load_data(i)

    columns = ['head', 'relation', 'tail']
    #test_score = model.predict(test[columns])
    # Call the predict method with the filtered test set
    #test_score = model.predict(test[columns])
    test_label = test['label'].values


    re_train_all = train[columns]
    #print("re_train_all",re_train_all)
    re_test_all = test[columns]
    #print("re_test_all",re_test_all)

    train_label = train['label']
    train_label_fs = train['label'].values
    test_label_fs = test['label'].values

    train_dense_features, test_dense_features = get_scaled_embeddings(
        re_train_all, re_test_all, False, n_components)


    train_des = get_features(re_train_all, fp_df, prodes_df, use_pro)
    test_des = get_features(re_test_all, fp_df, prodes_df, use_pro)
    #print("test_des",test_des)
    #print(np.isnan(test_des))


    feature_columns, train_model_input, test_model_input = get_attention_input(re_train_all, re_test_all,
                                                                         train_dense_features, test_dense_features,
                                                                         train_des, test_des,train_label_fs, test_label_fs,
                                                                         embedding_dim, n_components)
    
    #print(feature_columns.shape)

    roc_fpr_nfm, roc_tpr_nfm, roc_threshold_nfm, roc_a_nfm, pr_precision_nfm, pr_recall_nfm, pr_threshold_nfm, pr_a_nfm, pred_y = train_attention(
        feature_columns, train_model_input, train_label, test_model_input, test_label, patience)
    return roc_fpr_nfm, roc_tpr_nfm, roc_threshold_nfm, roc_a_nfm, pr_precision_nfm, pr_recall_nfm, pr_threshold_nfm, pr_a_nfm, re_train_all, train_label, re_test_all, test_label, pred_y


#train and test
################################################################
optimizer_params = {
    'optimizer': 'adam',
    'lr': 1e-4,
    'decay': 1e-5
}

ROC_AUC = []
PR_AUC = []
for i in range(10):
    print(i)
    roc_fpr, roc_tpr, roc_threshold, roc_a, pr_precision, pr_recall, pr_threshold, pr_a, re_train_all, train_label, re_test_all, test_label, pred_y = train(
        i, 50, 200, True, 25, optimizer_params)


    roc_curve = pd.DataFrame()
    roc_curve['roc_fpr'] = roc_fpr
    roc_curve['roc_tpr'] = roc_tpr
    roc_curve['roc_threshold'] = roc_threshold
    print(roc_curve)

    pr_curve = pd.DataFrame()
    pr_curve['pr_precision'] = pr_precision
    pr_curve['pr_recall'] = pr_recall
    pr_curve['pr_threshold'] = np.append(
        pr_threshold, values=np.nan)
    print(pr_curve)


    print('roc_auc: %f' % roc_a)
    print('pr_auc: %f' % pr_a)

    ROC_AUC.append(roc_a)
    PR_AUC.append(pr_a)


stable_metrics = pd.DataFrame()
stable_metrics['roc_auc'] = ROC_AUC
stable_metrics['pr_auc'] = PR_AUC
print(stable_metrics)
print(stable_metrics.describe())
