import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings

import jieba
import re


# 停用词
# 创建停用词列表
def get_stopwords_list():
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    stopwords.append('（')
    stopwords.append('）')
    return stopwords


# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.lcut(sentence.strip())
    return sentence_depart


def move_stopwords(sentence_list, stopwords_list):
    # 去停用词
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            if word != '\t':
                out_list.append(word)
    return ' '.join(out_list)


stopwords = get_stopwords_list()


def get_cut_list(x):
    sentence_depart = seg_depart(x)
    sentence_depart = move_stopwords(sentence_depart, stopwords)
    return sentence_depart


warnings.filterwarnings('ignore')

base = pd.read_csv('./data/train/base_info.csv')
label = pd.read_csv('./data/train/entprise_info.csv')
base = pd.merge(base, label, on=['id'], how='left')

base['oploc_list'] = base['oploc'].apply(lambda x: ' '.join([x[16 * i:16 * (i + 1)] for i in range(int(len(x) / 16))]))
base['dom_list'] = base['dom'].apply(lambda x: ' '.join([x[16 * i:16 * (i + 1)] for i in range(int(len(x) / 16))]))
base['opscope_word_list'] = base['opscope'].apply(get_cut_list)

oploc__tfidf_vector = TfidfVectorizer(min_df=30).fit(
    base['oploc_list'].tolist())
dom__tfidf_vector = TfidfVectorizer(min_df=30).fit(
    base['dom_list'].tolist())
opscope_tfidf_vector = TfidfVectorizer(min_df=30).fit(
    base['opscope_word_list'].tolist())

data = base[['id', 'oploc_list', 'dom_list', 'opscope_word_list', 'label']]
X_train = data[~data['label'].isnull()]
X_test = data[data['label'].isnull()]


def create_csr_mat_input(oploc_list, dom_list, opscope_word_list):
    return sp.hstack((oploc__tfidf_vector.transform(oploc_list),
                      dom__tfidf_vector.transform(dom_list),
                      opscope_tfidf_vector.transform(opscope_word_list)),
                     format='csr')


def kfold_tfidf_model(df_train, df_submit, target, model_list):
    for model in model_list:
        df_submit[f'{model.__class__.__name__}_pred'] = 0

    submit_x = create_csr_mat_input(df_submit['oploc_list'],
                                    df_submit['dom_list'],
                                    df_submit['opscope_word_list'])

    kfout = StratifiedKFold(n_splits=5, random_state=2020)
    kfin = StratifiedKFold(n_splits=5, random_state=2020)
    for model in tqdm(model_list):
        y_submit = 0
        for fold_, (trn_idx, tst_idx) in enumerate(kfout.split(df_train, target)):
            X, tst_x = df_train.iloc[trn_idx], df_train.iloc[tst_idx]
            y, tst_y = target.iloc[trn_idx], target.iloc[tst_idx]
            tst_x = create_csr_mat_input(tst_x['oploc_list'], tst_x['dom_list'], tst_x['opscope_word_list'])
            y_test_hat = 0
            for trn_idx, val_idx in kfin.split(X, y):
                tr_x, tr_y = create_csr_mat_input(
                    X['oploc_list'].iloc[trn_idx], X['dom_list'].
                        iloc[trn_idx], X['opscope_word_list'].iloc[trn_idx]), y.iloc[trn_idx]
                vl_x, vl_y = create_csr_mat_input(
                    X['oploc_list'].iloc[val_idx], X['dom_list'].
                        iloc[val_idx], X['opscope_word_list'].iloc[val_idx]), y.iloc[val_idx]

                if 'LGBM' in model.__class__.__name__ or 'XGB' in model.__class__.__name__:
                    model.fit(tr_x,
                              tr_y,
                              eval_set=[(vl_x, vl_y)],
                              early_stopping_rounds=400,
                              verbose=99999)
                else:
                    model.fit(tr_x, tr_y)

                y_test_hat += model.predict_proba(tst_x)[:, 1] / kfin.n_splits
                y_submit += model.predict_proba(submit_x)[:, 1] / (kfin.n_splits * kfout.n_splits)
            print(
                f'fold {fold_ + 1} model: {model.__class__.__name__} f1:{f1_score(tst_y, np.where(y_test_hat >= 0.5, 1, 0))}'
            )
            df_train.loc[df_train.iloc[tst_idx, :].
                             index, f'{model.__class__.__name__}_pred'] = y_test_hat

        f1 = f1_score(target, np.where(df_train[f'{model.__class__.__name__}_pred'] >= 0.5, 1, 0))
        print(f'model {model.__class__.__name__} all f1: {f1}')
        df_submit[f'{model.__class__.__name__}_pred'] += y_submit
    return pd.concat([df_train, df_submit])


estimators = [
    # RandomForestClassifier(n_estimators=300,
    #                        max_depth=20,
    #                        min_samples_split=20,
    #                        random_state=42,
    #                        n_jobs=-1),
    # GradientBoostingClassifier(n_estimators=350,
    #                            learning_rate=0.1,
    #                            random_state=42,
    #                            max_features='auto',
    #                            n_iter_no_change=50),
    SGDClassifier(loss="log", random_state=42, early_stopping=True),
    # LGBMClassifier(n_estimators=9999,
    #                learning_rate=0.05,
    #                n_jobs=-1,
    #                random_state=42,
    #                colsample_bytree=0.8,
    #                subsample=0.8,
    #                silent=True),
    # XGBClassifier(
    #     n_estimators=9999,
    #     learning_rate=0.05,
    #     colsample_bytree=0.8,
    #     max_depth=10,
    #     subsample=0.8,
    #     random_state=42
    #     # tree_method='gpu_hist',
    #     # predictor='gpu_predictor',
    #     # gpu_id=0
    # ),
]

# new = kfold_tfidf_model(X_train, X_test, X_train['label'], estimators)
# c = [f for f in new.columns if 'pred' in f or f == 'id']
# new[c].to_csv('tfidf_prob.csv', index=False)

tfidf_input = create_csr_mat_input(data['oploc_list'], data['dom_list'], data['opscope_word_list'])
result = pd.DataFrame({'id': data['id']})

lda = LatentDirichletAllocation(n_jobs=-1,
                                random_state=2020,
                                n_components=16)
result[[
    f'lda_{i + 1}' for i in range(lda.n_components)
]] = pd.DataFrame(lda.fit_transform(
    tfidf_input), index=result.index)

nmf = NMF(random_state=2020, n_components=16)
result[[
    f'nmf_{i + 1}' for i in range(nmf.n_components)
]] = pd.DataFrame(nmf.fit_transform(
    tfidf_input),
    index=result.index)

svd = TruncatedSVD(random_state=2020,
                   n_components=32)
result[[
    f'svd_{i + 1}' for i in range(svd.n_components)
]] = pd.DataFrame(svd.fit_transform(
    tfidf_input),
    index=result.index)

result.to_csv('tfidf_decomposition.csv', index=False)
