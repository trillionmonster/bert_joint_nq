import tqdm

from model import mk_model_for_test
import bert_utils
import tensorflow as tf
import numpy as np
from data import get_test_data
import json

import pandas as pd

def create_short_answer(entry):
    # if entry["short_answer_score"] < 1.5:
    #     return ""

    answer = []
    for short_answer in entry["short_answers"]:
        if short_answer["start_token"] > -1:
            answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))
    if entry["yes_no_answer"] != "NONE":
        answer.append(entry["yes_no_answer"])
    return " ".join(answer)


def create_long_answer(entry):

    answer = []
    if entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
    return " ".join(answer)

ds,token_map_ds = get_test_data(batch_size=1)
model = mk_model_for_test("",)
result=model.predict_generator(ds,verbose= 0)
np.savez_compressed('bert-joint-baseline-output.npz',
                    **dict(zip(['uniqe_id','start_logits','end_logits','answer_type_logits'],
                               result)))

all_results = [bert_utils.RawResult(*x) for x in zip(*result)]

print("Going to candidates file")

candidates_dict = bert_utils.read_candidates('../input/tensorflow2-question-answering/simplified-nq-test.jsonl')

print("setting up eval features")

eval_features = list(token_map_ds)

print("compute_pred_dict")

nq_pred_dict = bert_utils.compute_pred_dict(candidates_dict,
                                            eval_features,
                                            all_results,
                                            tqdm=None)

predictions_json = {"predictions": list(nq_pred_dict.values())}

print("writing json")

with tf.io.gfile.GFile('predictions.json', "w") as f:
    json.dump(predictions_json, f, indent=4)
test_answers_df = pd.read_json("/kaggle/input/predictions/predictions.json")
for var_name in ['long_answer_score', 'short_answer_score', 'answer_type']:
    test_answers_df[var_name] = test_answers_df['predictions'].apply(lambda q: q[var_name])
test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)
test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)
test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))

long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")
long_answers = dict([item for item in long_answers.items() if item[1] != ""])
short_answers= dict([item for item in short_answers.items() if item[1] != ""])

for item in long_answers.items():
    sample_submission.loc[sample_submission["example_id"] == str(item[0])+"_long","PredictionString"] = item[1]
for item in short_answers.items():
    sample_submission.loc[sample_submission["example_id"] == str(item[0])+"_short","PredictionString"] = item[1]

sample_submission.to_csv("submission.csv",index = False)
