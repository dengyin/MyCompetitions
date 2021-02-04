import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# load data
annual_report_info = pd.read_csv('./data/train/annual_report_info.csv')
base_info = pd.read_csv('./data/train/base_info.csv')
change_info = pd.read_csv('./data/train/change_info.csv')
entprise_info = pd.read_csv('./data/train/entprise_info.csv')
news_info = pd.read_csv('./data/train/news_info.csv')
other_info = pd.read_csv('./data/train/other_info.csv')
tax_info = pd.read_csv('./data/train/tax_info.csv')
entprise_evaluate = pd.read_csv('./data/entprise_evaluate.csv')

data = pd.concat([entprise_info, entprise_evaluate.rename(columns={'score': 'label'})], ignore_index=True)


# extract features

def kfold_mean(df_train, df_test, target, target_mean_list):
    folds = StratifiedKFold(n_splits=5, random_state=2020)

    mean_of_target = df_train[target].mean()

    for fold_, (trn_idx, val_idx) in tqdm(
            enumerate(folds.split(df_train, y=df_train['label']))):
        tr_x = df_train.iloc[trn_idx, :]
        vl_x = df_train.iloc[val_idx, :]

        for col in target_mean_list:
            df_train.loc[vl_x.index, f'{col}_target_enc'] = vl_x[col].map(
                tr_x.groupby(col)[target].mean())

    for col in target_mean_list:
        df_train[f'{col}_target_enc'].fillna(mean_of_target, inplace=True)

        df_test[f'{col}_target_enc'] = df_test[col].map(
            df_train.groupby(col)[f'{col}_target_enc'].mean())

        df_test[f'{col}_target_enc'].fillna(mean_of_target, inplace=True)
    return pd.concat([df_train, df_test], ignore_index=True)


def extract_base_info_info(data):
    data['district_FLAG1'] = (data['orgid'].fillna('').apply(lambda x: str(x)[:6]) == \
                              data['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
    data['district_FLAG2'] = (data['orgid'].fillna('').apply(lambda x: str(x)[:6]) == \
                              data['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)
    data['district_FLAG3'] = (data['oplocdistrict'].fillna('').apply(lambda x: str(x)[:6]) == \
                              data['jobid'].fillna('').apply(lambda x: str(x)[:6])).astype(int)

    data['person_SUM'] = data[['empnum', 'parnum', 'exenum']].sum(1)
    data['person_NULL_SUM'] = data[['empnum', 'parnum', 'exenum']].isnull().astype(int).sum(1)

    data['empnum'] = data['empnum'].fillna(-1)
    data['compform'] = data['compform'].fillna(-1)
    data['parnum'] = data['parnum'].fillna(-1)
    data['exenum'] = data['exenum'].fillna(-1)
    data['opform'] = data['opform'].fillna('-1')
    data['venind'] = data['venind'].fillna(-1)
    data['reccap'] = data['reccap'].fillna(-1)
    data['regcap'] = data['regcap'].fillna(-1)
    data['venind_isna'] = (data['venind'] < 0).astype(np.int8)
    data['reccap_isna'] = (data['reccap'] < 0).astype(np.int8)
    data['regcap_isna'] = (data['regcap'] < 0).astype(np.int8)
    data['enttypeitem'] = data['enttypeitem'].fillna(-1)
    data['enttypeminu'] = data['enttypeminu'].fillna(-1)

    # 未缺失数目
    data['not_na_count'] = 0
    for i in range(len(data)):
        if not math.isnan(data['congro'][i]):
            data['not_na_count'][i] += 1
        if not math.isnan(data['protype'][i]):
            data['not_na_count'][i] += 1
        if not math.isnan(data['forreccap'][i]):
            data['not_na_count'][i] += 1
        if not math.isnan(data['forregcap'][i]):
            data['not_na_count'][i] += 1

    drop = ['ptbusscope', 'midpreindcode',
            'protype', 'forreccap',
            'forregcap', 'congro']
    for f in drop:
        del data[f]

    data['opto'] = data['opto'].fillna('2099-99-99')
    data['opto_isna'] = (data['opto'].apply(lambda x: int(x.split('-')[1]) == 99)).astype(np.int8)
    data['year_to'] = data['opto'].apply(lambda x: int(x.split('-')[0]))
    # data['month_to'] = data['opto'].apply(lambda x: int(x.split('-')[1]))
    data['year_from'] = data['opfrom'].apply(lambda x: int(x.split('-')[0]))
    # data['month_from'] = data['opfrom'].apply(lambda x: int(x.split('-')[1]))
    data['year_range'] = data['year_to'] - data['year_from']
    # del data['year_from'], data['year_to']
    del data['opfrom']
    del data['opto']
    data['cap_diff'] = data['reccap'] - data['regcap']

    # data['opscope_all_len'] = data['opscope'].apply(lambda x: len(x))
    # data['opscope'] = data['opscope'].apply(lambda x: re.sub(u"\\（.*?\\）", "", x))
    # data['opscope'] = data['opscope'].apply(lambda x: x.replace('*', ''))
    # data['opscope_len'] = data['opscope'].apply(lambda x: len(x))
    # data['opscope_len_diff'] = data['opscope_all_len'] - data['opscope_len']
    # data['opscope'] = data['opscope'].apply(lambda x: re.split('[，。、；]', x))
    # data['opscope_count'] = data['opscope'].apply(lambda x: len(x))
    # data['opscope_lcratio'] = data['opscope_len'] / data['opscope_count']

    data['dom_len'] = data['dom'].apply(lambda x: len(x) / 16)
    data['oploc_len'] = data['oploc'].apply(lambda x: len(x) / 16)
    data['oploc'] = data['oploc'].apply(lambda x: x[:16])
    data['len_diff'] = data.apply(lambda x: x['oploc_len'] - x['dom_len'], axis=1)
    data['opscope_legal'] = data['opscope'].apply(lambda x: 1 if '依法须经批准的项目' in x else 0)
    del data['dom'], data['opscope']
    # del data['oploc']

    lb = LabelEncoder()
    data['industryphy'] = lb.fit_transform(data['industryphy'])
    data['opform'] = lb.fit_transform(data['opform'])
    data['oploc'] = lb.fit_transform(data['oploc'])
    data['industryco'] = lb.fit_transform(data['industryco'])

    # count encode
    for col in ['oplocdistrict', 'industryphy', 'enttype', 'state', 'orgid',
                'adbusign', 'townsign', 'oploc', 'regtype', 'enttypegb',
                'enttypeitem', 'enttypeminu', 'jobid', 'industryco']:
        data[col + '_COUNT'] = data[col].map(data[col].value_counts())
        col_idx = data[col].value_counts()
        for idx in col_idx[col_idx < 10].index:
            data[col] = data[col].replace(idx, -1)

    data = kfold_mean(data[~data['label'].isna()], data[data['label'].isna()], 'label',
                      ['oplocdistrict', 'industryphy', 'enttype', 'state', 'orgid',
                       'adbusign', 'townsign', 'oploc', 'regtype', 'enttypegb',
                       'enttypeitem', 'enttypeminu', 'jobid', 'industryco'])

    return data


def extract_annual_report_info(annual_report_):
    annual_report = annual_report_.copy().sort_values(by=['id', 'ANCHEYEAR'])
    result = annual_report.groupby('id').agg(
        {
            'ANCHEYEAR': ['nunique', lambda x: x.diff().max(), lambda x: x.diff().mean()],
            'EMPNUM': ['mean', 'std', lambda x: x.diff().mean(), lambda x: x.diff().std()],
            'COLGRANUM': ['mean', 'std'],
            'RETSOLNUM': ['mean', 'std'],
            'DISPERNUM': ['mean', 'std'],
            'UNENUM': ['mean', 'std'],
            'COLEMPLNUM': ['mean', 'std'],
            'RETEMPLNUM': ['mean', 'std'],
            'DISEMPLNUM': ['mean', 'std'],
            'UNEEMPLNUM': ['mean', 'std']
        }
    ).reset_index()
    result.columns = ['id',
                      'year_nunique',
                      'year_diff_max',
                      'year_diff_mean',
                      'EMPNUM_mean',
                      'EMPNUM_std',
                      'EMPNUM_diff_mean',
                      'EMPNUM_diff_std',
                      'COLGRANUM_mean',
                      'COLGRANUM_std',
                      'RETSOLNUM_mean',
                      'RETSOLNUM_std',
                      'DISPERNUM_mean',
                      'DISPERNUM_std',
                      'UNENUM_mean',
                      'UNENUM_std',
                      'COLEMPLNUM_mean',
                      'COLEMPLNUM_std',
                      'RETEMPLNUM_mean',
                      'RETEMPLNUM_std',
                      'DISEMPLNUM_mean',
                      'DISEMPLNUM_std',
                      'UNEEMPLNUM_mean',
                      'UNEEMPLNUM_std']

    for f in ['STATE', 'BUSSTNAME', 'PUBSTATE', 'EMPNUMSIGN', 'WEBSITSIGN', 'FORINVESTSIGN', 'STOCKTRANSIGN']:
        result[[f'{f}_count_vec_{i}' for i in range(annual_report[f].nunique(dropna=False))]] = pd.DataFrame(
            CountVectorizer(vocabulary=annual_report[f].unique().astype(str)).fit_transform(
                annual_report.groupby('id')[f].apply(lambda x: ' '.join(list(x.astype(str))))).toarray(),
            index=result.index)
    return result


def extract_change_info(change):
    result = change.groupby('id').agg(
        {
            'bgxmdm': ['nunique', lambda x: x.nunique() / len(x)],
            'bgq': ['nunique', lambda x: x.nunique() / len(x)],
            'bgh': ['nunique', lambda x: x.nunique() / len(x)]
        }
    ).reset_index()
    result.columns = ['id', 'bgxmdm_nunique', 'bgxmdm_nunique_rto', 'bgq_nunique', 'bgq_nunique_rto', 'bgh_nunique',
                      'bgh_nunique_rto']
    return result


def extract_other_info(others):
    others['other_SUM'] = others[['legal_judgment_num', 'brand_num', 'patent_num']].sum(1)
    others['other_NULL_SUM'] = others[['legal_judgment_num', 'brand_num', 'patent_num']].isnull().astype(
        int).sum(1)
    result = others.groupby('id').agg(
        {
            'other_SUM': ['mean'],
            'other_NULL_SUM': ['mean']
        }
    ).reset_index()
    result.columns = ['id',
                      'other_SUM',
                      'other_NULL_SUM']
    return result


def extract_news_info(news):
    news_info['public_date'] = news_info['public_date'].apply(lambda x: x if '-' in str(x) else np.nan)
    news_info['public_date'] = pd.to_datetime(news_info['public_date'])
    news_info['public_date'] = (datetime.now() - news_info['public_date']).dt.days

    result = news.groupby('id').agg({'public_date': ['count']}).reset_index()
    result.columns = ['id', 'public_date_COUNT']
    for f in ['positive_negtive']:
        result[[f'{f}_count_vec_{i}' for i in range(news[f].nunique(dropna=False))]] = pd.DataFrame(
            CountVectorizer(vocabulary=news[f].unique().astype(str)).fit_transform(
                news.groupby('id')[f].apply(lambda x: ' '.join(list(x.astype(str))))).toarray(),
            index=result.index)
    return result


def extract_tax_info(tax):
    tax['income'] = tax['TAX_AMOUNT'] / tax['TAX_RATE']
    result = tax.groupby('id').agg(
        {
            'TAX_CATEGORIES': ['count'],
            'TAX_ITEMS': ['count'],
            'TAXATION_BASIS': ['mean', 'std', 'count'],
            'TAX_RATE': ['mean', 'std'],
            'TAX_AMOUNT': ['mean', 'std', 'max', 'min'],
        }
    ).reset_index()
    result.columns = ['id',
                      'TAX_CATEGORIES_count',
                      'TAX_ITEMS_count',
                      'TAXATION_BASIS_mean',
                      'TAXATION_BASIS_std',
                      'TAXATION_BASIS_count',
                      'TAX_RATE_mean',
                      'TAX_RATE_std',
                      'TAX_AMOUNT_mean',
                      'TAX_AMOUNT_std',
                      'TAX_AMOUNT_max',
                      'TAX_AMOUNT_min',
                      ]

    for f in ['TAX_CATEGORIES']:
        result[[f'{f}_count_vec_{i}' for i in range(tax[f].nunique(dropna=False))]] = pd.DataFrame(
            CountVectorizer(vocabulary=tax[f].unique().astype(str)).fit_transform(
                tax.groupby('id')[f].apply(lambda x: ' '.join(list(x.astype(str))))).toarray(),
            index=result.index)

    # tax_items_tfidf
    tax_items_tfidf = TfidfVectorizer(vocabulary=tax['TAX_ITEMS'].unique().astype(str), min_df=20).fit_transform(
        tax.groupby('id').apply(lambda x: ' '.join(list(x))).tolist())

    lda = LatentDirichletAllocation(n_jobs=-1,
                                    random_state=2020,
                                    n_components=8)
    result[[
        f'lda_tax_items{i + 1}' for i in range(lda.n_components)
    ]] = pd.DataFrame(lda.fit_transform(
        tax_items_tfidf),
        index=result.index)

    nmf = NMF(random_state=2020, n_components=8)
    result[[
        f'nmf_tax_items{i + 1}' for i in range(nmf.n_components)
    ]] = pd.DataFrame(nmf.fit_transform(
        tax_items_tfidf),
        index=result.index)

    svd = TruncatedSVD(random_state=2020,
                       n_components=8)
    result[[
        f'svd_tax_items{i + 1}' for i in range(svd.n_components)
    ]] = pd.DataFrame(svd.fit_transform(
        tax_items_tfidf),
        index=result.index)

    return result


data = pd.merge(left=data, right=base_info, how='left', on='id')
data = extract_base_info_info(data)

data = data.merge(pd.read_csv('tfidf_decomposition.csv'), how='left', on='id')


# data = data.merge(pd.read_csv('tfidf_decomposition2.csv'), how='left', on='id')


# data = data.merge(pd.read_csv('opscope_vec.csv'), how='left', on='id')

# data = data.merge(pd.read_csv('opscope_bert_vec.csv'), how='left', on='id')


tfidf_data = pd.read_csv('tfidf_prob.csv')
tfidf_data = tfidf_data[[f for f in tfidf_data.columns if 'pred' in f or f == 'id']]
data = data.merge(tfidf_data, how='left', on='id')


def combine_info(data, info_data, extract_func):
    info = extract_func(info_data)
    data = data.merge(info, on='id', how='left')
    return data


for info_data, extract_func in tqdm([
    (annual_report_info, extract_annual_report_info),
    (change_info, extract_change_info),
    (tax_info, extract_tax_info),
    # (other_info, extract_other_info),
    (news_info, extract_news_info),
], desc='extract features'):
    data = combine_info(data, info_data, extract_func)


def filter_col_by_nan(df, ratio=0.05):
    cols = []
    for col in df.columns:
        if df[col].isna().mean() >= (1 - ratio):
            cols.append(col)
    return cols


drop_columns = ['id', 'label']
cat_features = ['oplocdistrict', 'industryphy', 'enttype', 'state',
                'opto_isna', 'industryco', 'oploc', 'orgid', 'adbusign',
                'townsign', 'regtype', 'enttypegb', 'compform', 'jobid',
                'venind', 'enttypeitem', 'enttypeminu', 'opform',
                'opscope_legal', 'regcap_isna', 'reccap_isna', 'venind_isna']
data[cat_features] = data[cat_features].astype('category')
num_features = [f for f in data.columns if f not in drop_columns + cat_features]
data = data.drop(filter_col_by_nan(data[num_features], 0.01), axis=1)
data[num_features] = data[num_features].fillna(-1)
features = num_features
train, X_submit = data[~data['label'].isna()], data[data['label'].isna()]

train['sample_weight'] = 1


# train & predict
def my_metric(y_true, y_pred, sample_weight=None):
    f1 = f1_score(y_true, np.where(y_pred >= 0.5, 1, 0), sample_weight=sample_weight)
    p = precision_score(y_true, np.where(y_pred >= 0.5, 1, 0), sample_weight=sample_weight)
    r = recall_score(y_true, np.where(y_pred >= 0.5, 1, 0), sample_weight=sample_weight)
    return 0.5 * p + 0.3 * r + 0.2 * f1


def kfold_xgb(train, X_submit, target, para, seed=2020):
    kfout = StratifiedKFold(n_splits=5, random_state=seed)
    kfin = StratifiedKFold(n_splits=5, random_state=seed)
    y_pred = target - target
    y_submit = 0

    for fold, (train_index, test_index) in tqdm(enumerate(kfout.split(train, target))):
        X, X_test = train.iloc[train_index], train.iloc[test_index]
        y, y_test = target.iloc[train_index], target.iloc[test_index]
        y_test_hat = 0
        for train_index, val_index in kfin.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = xgb.XGBClassifier(n_estimators=9999,
                                      random_state=seed,
                                      n_jobs=-1,
                                      **para
                                      )
            model.fit(X_train[features], y_train, sample_weight=X_train['sample_weight'],
                      eval_set=[(X_train[features], y_train), (X_val[features], y_val)],
                      eval_metric=['auc'], sample_weight_eval_set=[X_train['sample_weight'], X_val['sample_weight']],
                      early_stopping_rounds=400, verbose=400)
            y_test_hat += model.predict_proba(X_test[features])[:, 1] / kfin.n_splits
            y_submit += model.predict_proba(X_submit[features])[:, 1] / (kfin.n_splits * kfout.n_splits)

        fold_f1 = my_metric(y_test, np.where(y_test_hat >= 0.5, 1, 0), sample_weight=X_test['sample_weight'])
        print(f'{fold + 1}fold f1: {fold_f1}')
        y_pred.loc[y_test.index] = y_test_hat

    f1 = my_metric(target, np.where(y_pred >= 0.5, 1, 0), sample_weight=train['sample_weight'])
    auc = roc_auc_score(target, y_pred, sample_weight=train['sample_weight'])
    print(f'all f1: {f1}, auc: {auc}')
    # xgb.plot_importance(model,
    #                     max_num_features=30,
    #                     height=0.5,
    #                     )
    # plt.show()
    return y_pred, y_submit


y_pred_all_para = []
y_submit_all_para = []
paras = [
    {'max_depth': 10,
     'learning_rate': 0.03727181634370282,
     'colsample_bytree': 0.949223410342457,
     'min_child_weight': 0,
     'reg_alpha': 0.6625276325552516,
     'reg_lambda': 1.2301551691940324,
     'subsample': 0.9960414508457138},
    {'max_depth': 17,
     'learning_rate': 0.045709423656353165,
     'colsample_bytree': 0.850602022977941,
     'min_child_weight': 0,
     'reg_alpha': 0.7422489500301994,
     'reg_lambda': 0.13311530398928895,
     'subsample': 0.9972532955191347},
    {'max_depth': 11,
     'learning_rate': 0.03324,
     'colsample_bytree': 0.8578,
     'min_child_weight': 0,
     'reg_alpha': 0.1414,
     'reg_lambda': 0.898,
     'subsample': 0.9938}
]
for i, para in tqdm(enumerate(paras)):
    y_pred = 0
    y_submit = 0
    seeds = [2020]
    min_child_weight = np.random.randint(0, 5)
    for seed in seeds:
        y_pred_seed, y_submit_seed = kfold_xgb(train, X_submit, train['label'], para, seed)
        y_pred += y_pred_seed / len(seeds)
        y_submit += y_submit_seed / len(seeds)

    f1 = my_metric(train['label'], np.where(y_pred >= 0.5, 1, 0), sample_weight=train['sample_weight'])
    auc = roc_auc_score(train['label'], y_pred, sample_weight=train['sample_weight'])
    print(f'round {i + 1},  all seed f1: {f1}, all seed auc: {auc}')

    y_pred_all_para.append(y_pred)
    y_submit_all_para.append(y_submit)
# 融合
# mean
y_pred_all_para_mean = np.column_stack(y_pred_all_para).mean(axis=1)
y_submit_all_para_mean = np.column_stack(y_submit_all_para).mean(axis=1)
f1 = my_metric(train['label'], np.where(y_pred_all_para_mean >= 0.5, 1, 0), sample_weight=train['sample_weight'])
auc = roc_auc_score(train['label'], y_pred_all_para_mean, sample_weight=train['sample_weight'])
print(f'all para mean f1: {f1}, all para mean auc: {auc}')

# min
y_pred_all_para_min = np.where(np.column_stack(y_pred_all_para) >= 0.5, 1, 0).min(axis=1)
y_submit_all_para_min = np.where(np.column_stack(y_submit_all_para) >= 0.5, 1, 0).min(axis=1)
f1 = my_metric(train['label'], y_pred_all_para_min, sample_weight=train['sample_weight'])
print(f'all para min f1: {f1}')

# Geometric mean
from functools import reduce
y_pred_all_para_geo_mean = reduce(lambda x, y: x * y, y_pred_all_para) ** (1 / len(y_pred_all_para))
y_submit_all_para_geo_mean = reduce(lambda x, y: x * y, y_submit_all_para) ** (1 / len(y_submit_all_para))
f1 = my_metric(train['label'], np.where(y_pred_all_para_geo_mean >= 0.5, 1, 0), sample_weight=train['sample_weight'])
auc = roc_auc_score(train['label'], y_pred_all_para_geo_mean, sample_weight=train['sample_weight'])
print(f'all para geo mean f1: {f1}, all para mean geo auc: {auc}')

# # submit
# submit = pd.read_csv('./data/entprise_submit.csv')
# submit['score'] = submit['id'].map(pd.Series(np.where(y_submit_all_para_mean >= 0.5, 1, 0), index=X_submit['id']))
# # submit['score'] = submit['id'].map(pd.Series(y_submit_all_para_min, index=X_submit['id']))
# print(submit['score'].mean())
# submit.to_csv('./submit/submit_xgb.csv', index=False)

# all f1: 0.8517172044357404, auc: 0.9931261675669467 mean: 0.0886 lb:0.850
