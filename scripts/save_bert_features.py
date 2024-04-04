import sys, os
cwd = os.getcwd()
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
import numpy as np
import gzip, json, os
from tqdm import trange, tqdm
import argparse

def load_data(data_type='train', role='guide', instr_type='full'):
    if instr_type == 'full':
        data_loc = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}.json.gz'
    elif instr_type == 'subtask':
        data_loc = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_subtasks_rollouts_merged_hli.json.gz'
    elif instr_type == 'high_level':
        data_loc = f'{cwd}/../data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_{role}_high_level_instr.json.gz'
    else:
        raise NotImplementedError(f" Intruction type: {instr_type} not defined")
    
    data = {}
    with gzip.open(data_loc,"rt",) as f:
        data.update(json.load(f))
    return data

def get_save_dir(data_type='train', role='guide', instr_type='full'):
    if instr_type == 'full':
        save_dir = f'{cwd}/../data/datasets/RxR_VLNCE_v0/text_features/rxr_{data_type}/'
    elif instr_type == 'subtask':
        save_dir = f'{cwd}/../data/datasets/RxR_VLNCE_v0/text_features_subtasks_merged/rxr_{data_type}/'
    elif instr_type == 'high_level':
        save_dir = f'{cwd}/../data/datasets/RxR_VLNCE_v0/text_features_highlevel_instr/rxr_{data_type}/'
    else:
        raise NotImplementedError(f" Intruction type: {instr_type} not defined")

    return save_dir

def make_bert_preprocess_model(seq_length=128):
    input_segments = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.load(f'{cwd}/../data/bert_models/bert_multi_cased_preprocess_3/')

    tokenizer = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')
    tokenized_input = [tokenizer(input_segments)]

    packer = hub.KerasLayer(preprocessor.bert_pack_inputs,
                        arguments=dict(seq_length=seq_length),
                        name='packer')
    model_inputs = packer(tokenized_input)
    return tf.keras.Model(input_segments, model_inputs)

def load_model():
    preprocess_model = make_bert_preprocess_model()
    encoder = hub.KerasLayer(f'{cwd}/../data/bert_models/bert_multi_cased_L-12_H-768_A-12_4/',trainable=True)
    return preprocess_model, encoder

if __name__== "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instr_type",
        choices=["full", "subtask", "high_level"],
        required=True,
        help="Instruction type (full, subtask, high_level)",
    )

    parser.add_argument(
        "--data_type",
        choices=["train", "val_seen", "val_unseen"],
        required=True,
        help="data type (train, val_seen, val_unseen)",
    )
    parser.add_argument(
        "--role",
        choices=["guide", "follower"],
        required=True,
        help="Role (guide, follower)",
    )

    args = parser.parse_args()

    data_type = args.data_type
    role = args.role
    instr_type = args.instr_type

    data = load_data(data_type=data_type, role=role, instr_type=instr_type)
    n_episodes = len(data['episodes'])

    save_dir = get_save_dir(data_type=data_type, role=role, instr_type=instr_type)
    os.makedirs(save_dir, exist_ok=True)

    save_file_name = "{id:06}_en_text_features.npz"

    preprocess_model, encoder = load_model()
    batch_size = 400
    n_batch=0
    done = False
    while not done:
        batch_end = min(n_episodes,(n_batch+1)*batch_size)
        print(f'batch:{n_batch}/{n_episodes/batch_size}')

        sentences = tf.constant([data['episodes'][i]['instruction']['instruction_text'] for i in range(n_batch*batch_size, batch_end)])

        preprocessed_input = preprocess_model(sentences)
        output = encoder(preprocessed_input)

        out_inx = 0
        for i in trange(n_batch*batch_size, batch_end):

            instruction_id = data['episodes'][i]['instruction']['instruction_id']
            file_name = save_dir + save_file_name.format(id=int(instruction_id))
            if os.path.exists(file_name):
                out_inx += 1
            else:
                np.savez(file_name, features=output['sequence_output'][out_inx])
                out_inx += 1


        if batch_end == n_episodes:
            done=True
            print(f"DONE SAVING ALL FEATURES FOR SUBTASK INSTRUCTIONS to {save_dir}!!")
        
        n_batch += 1

    