import json
import numpy as np
import pandas as pd

original_file = "../../data/test.original.json"
in_file = '../predict/predict_char_4000_1.csv'
out_file = "../predict/final_result.json"
query_dict = {}
with open(original_file, 'r') as ori_f:
    for line in ori_f:
        line = json.loads(line)
        query_id = line['query_id']
        query = line['query']
        query_dict[query_id] = query


df = pd.read_csv(in_file)

unique_query_ids = list(set(df['query_id']))
unique_query_ids.sort()
print("len_uniq:", len(unique_query_ids))

with open(out_file, 'w') as f:
    for query_id in unique_query_ids:
        if query_id == "query_id":
            continue
        passages = df[df['query_id'] == query_id]
        rank = range(0, passages.count()['label'])
        result = passages.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        result['rank'] = rank
        # result.drop('score', axis=1, inplace=True)
        ranklist = []
        for i in range(0, passages.count()['label']):
            rank = {"passage_id": result['passage_id'][i], "rank": result['rank'][i]}
            ranklist.append(rank)
        res = {"query_id": query_id, "query":query_dict[query_id],"ranklist":ranklist}
        f.write(str(res).replace('\'', '\"') + '\n')





