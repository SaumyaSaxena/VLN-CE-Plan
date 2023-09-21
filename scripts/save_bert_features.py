import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
import numpy as np
import gzip, json, os
from tqdm import trange


def load_data(data_type='train'):
    data_loc = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/{data_type}_guide_high_level_instr_143.json.gz'
    data_hl = {}
    with gzip.open(data_loc,"rt",) as f:
        data_hl.update(json.load(f))

    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))
    en_idx = [i for i in range(len(data['episodes'])) if ('en' in data['episodes'][i]['instruction']['language'])]

    import ipdb; ipdb.set_trace()

def make_bert_preprocess_model(seq_length=128):
    input_segments = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.load('/home/sax1rng/Projects/VLN-CE-Plan/data/bert_models/bert_multi_cased_preprocess_3/')

    tokenizer = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')
    tokenized_input = [tokenizer(input_segments)]

    packer = hub.KerasLayer(preprocessor.bert_pack_inputs,
                        arguments=dict(seq_length=seq_length),
                        name='packer')
    model_inputs = packer(tokenized_input)
    return tf.keras.Model(input_segments, model_inputs)

def load_model():
    preprocess_model = make_bert_preprocess_model()
    encoder = hub.KerasLayer('/home/sax1rng/Projects/VLN-CE-Plan/data/bert_models/bert_multi_cased_L-12_H-768_A-12_4/',trainable=True)
    return preprocess_model, encoder

if __name__== "__main__":
    # load_data()
    data_type = 'val_unseen'
    save_dir = f'/fs/scratch/rng_cr_aas3_gpu_user_c_lf/sp02_025_rrt/datasets/rxr-data/text_features_highlevel_instr/rxr_{data_type}/'
    save_file_name = "{id:06}_en_text_features.npz"

    data_location = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz'
    data = {}
    with gzip.open(data_location,"rt",) as f:
        data.update(json.load(f))

    preprocess_model, encoder = load_model()

    for batch in trange(0,37):
        hl_instr = {}
        data_loc = f'/home/sax1rng/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/high_level_instr/{data_type}_guide_high_level_instr_{batch}.json.gz'
        with gzip.open(data_loc,"rt",) as f:
            hl_instr.update(json.load(f))
        
        sentences = tf.constant(list(hl_instr.values()))

        preprocessed_input = preprocess_model(sentences)
        output = encoder(preprocessed_input)
        print("Saving files")
        for i, idx in enumerate(hl_instr.keys()):
            instruction_id = data['episodes'][int(idx)]['instruction']['instruction_id']
            file_name = save_dir + save_file_name.format(id=int(instruction_id))
            if os.path.exists(file_name):
                continue
            else:
                np.savez(file_name, features=output['sequence_output'][i])


    