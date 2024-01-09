# Data collection for sub-instructions/sub-trajectories

## Rollout actions in the environments
This step is needed since the data ({data_type}_{role}_gt.json.gz) provided with the main VLN-CE repo does not include orientation information. This information is crutial for splitting the trajectories into sub-trajectories.

```
python scripts/rollout_train_actions_in_env.py
```
Change `SIMULATOR_GPU_IDS` and `NUM_ENVIRONMENTS` in `vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml` to rollout multiple environments in parallel. 

## Extract sub-instructions from full instructions
```
python scripts/save_subtask_llama_instructions.py
```
Above script saves data in batches.

## Extract sub-trajectories from rolled out full trajectories and sub-instructions
```
python scripts/save_subtask_episodes_rollout.py
```

## Merge trajectories with no forward steps
```
python scripts/reformat_data_to_combine_non_fwd_eps.py
```

## Save BERT features for the new instructions
Download the BERT model ([reference](https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/classify_text_with_bert.ipynb)):
```
wget "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4?tf-hub-format=compressed" -O bert_multi_cased_L-12_H-768_A-12_4.tar.gz
wget "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3?tf-hub-format=compressed" -O bert_multi_cased_preprocess_3.tar.gz
```

Extract the above files to `VLN-CE-Plan/data/bert_models`
```
mkdir -p VLN-CE-Plan/data/bert_models/bert_multi_cased_L-12_H-768_A-12_4
tar -xf bert_multi_cased_L-12_H-768_A-12_4.tar.gz -C VLN-CE-Plan/data/bert_models/bert_multi_cased_L-12_H-768_A-12_4


mkdir -p VLN-CE-Plan/data/bert_models/bert_multi_cased_preprocess_3
tar -xf bert_multi_cased_preprocess_3.tar.gz -C VLN-CE-Plan/data/bert_models/bert_multi_cased_preprocess_3
```

Finally, generate BERT for the sub-instructions:
```
python scripts/save_bert_features_subtasks.py
```
