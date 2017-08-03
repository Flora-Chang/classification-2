# encoding: utf-8
import numpy as np
import pandas as pd
from tester import dcg_k, normalized_dcg_k

file1 = "../predict/predict_char_4000_4.csv"
file2 = "../predict/char_predict_3.csv"
file3 = "../predict/dev_predict_char_query30_doc200.csv"
file_list = [file2]
df1 = pd.read_csv(file1)

df_new = df1.sort_values(by=['query_id', 'passage_id']).reset_index(drop=True)
print(df_new.head(10))
for file in file_list:
    df = pd.read_csv(file)
    df = df.sort_values(by=['query_id', 'passage_id']).reset_index(drop=True)
    print('+' * 20)
    print(df.head(10))
    print('+' * 20)
    df_new['score'] += df['score']
    print('-' * 20)
    print(df_new.head(10))

df_new.to_csv("../predict/" + "char_new_4.csv", index=False)
unique_query_ids = list(set(df_new['query_id']))
print("len_uniq:", len(unique_query_ids))

Norm_DCG3 = []
Norm_DCG5 = []
Norm_FULL =[]
DCG_3 = []
DCG_5 = []
DCG_full = []

zero_3 = 0
zero_5 = 0

for query_id in unique_query_ids:
    if query_id == "query_id":
        continue
    passages = df_new[df_new['query_id'] == query_id]
    rank = range(1, passages.count()['label'] + 1)

    # result = passages.sort(['score'], ascending=False).reset_index(drop=True)
    real = passages.sort_values(by=['label'], ascending=False).reset_index(drop=True)
    real['rank'] = rank
    real.drop('score', axis=1, inplace=True)
    result = passages.sort_values(by=['score'], ascending=False).reset_index(drop=True)
    result['rank'] = rank
    # result.drop('score', axis=1, inplace=True)

    dcg_3 = dcg_k(result, 3)
    dcg_5 = dcg_k(result, 5)
    dcg_full = dcg_k(result, rank[-1])
    norm_dcg_3 = normalized_dcg_k(result, real, 3)
    norm_dcg_5 = normalized_dcg_k(result, real, 5)
    norm_dcg_full = normalized_dcg_k(result, real, rank[-1])

    if dcg_5 < 0.1:
        zero_3 += 1
        zero_5 += 1
    elif dcg_3 < 0.1:
        zero_3 += 1

    Norm_DCG3.append(norm_dcg_3)
    Norm_DCG5.append(norm_dcg_5)
    Norm_FULL.append(norm_dcg_full)

    DCG_3.append(dcg_3)
    DCG_5.append(dcg_5)
    DCG_full.append(dcg_full)

dcg_3_mean = np.mean(DCG_3)
dcg_5_mean = np.mean(DCG_5)
dcg_full_mean = np.mean(DCG_full)

norm_dcg_3_mean = np.mean(Norm_DCG3)
norm_dcg_5_mean = np.mean(Norm_DCG5)
norm_dcg_full_mean = np.mean(Norm_FULL)

print(len(DCG_3))
print(len(DCG_5))

print("number of Zero DCG@3: ", zero_3)
print("number of Zero DCG@5: ", zero_5)
print("DCG@3 Mean: ", dcg_3_mean, "\tNorm: ", norm_dcg_3_mean)
print("DCG@5 Mean: ", dcg_5_mean, "\tNorm: ", norm_dcg_5_mean)
print("DCG@full Mean: ", dcg_full_mean, "\tNorm: ", norm_dcg_full_mean)
print("=" * 60)
