import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Registers the ops.
import numpy as np
import gzip, json

def make_bert_preprocess_model(seq_length=128):
    input_segments = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.load('/home/saumyas/Projects/VLN-CE-Plan/data/bert_models/bert_multi_cased_preprocess_3/')

    tok = preprocessor.tokenize(tf.constant(['Hello TensorFlow!']))

    tokenizer = hub.KerasLayer(preprocessor.tokenize, name='tokenizer')

    tokenized_input = [tokenizer(input_segments)]

    packer = hub.KerasLayer(preprocessor.bert_pack_inputs,
                        arguments=dict(seq_length=seq_length),
                        name='packer')
    model_inputs = packer(tokenized_input)
    return tf.keras.Model(input_segments, model_inputs)

def load_text_features(instruction_id):
    feature_file = "/home/saumyas/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/text_features/rxr_train/{id:06}_en_text_features.npz"
    feature_file = feature_file.format(id=int(instruction_id))
    return np.load(feature_file)

if __name__== "__main__":
    # Loading text instructions
    data_type = 'train'
    data_locations = [f'/home/saumyas/Projects/VLN-CE-Plan/data/datasets/RxR_VLNCE_v0/{data_type}/{data_type}_guide.json.gz']
    data = {}
    for data_file in data_locations:
        with gzip.open(data_file,"rt",) as f:
            data.update(json.load(f))
    
    episode_id = 100
    instruction_id = data['episodes'][episode_id]['instruction']['instruction_id']
    sentences = tf.constant([data['episodes'][episode_id]['instruction']['instruction_text']])
    print("Instruction to encode:", data['episodes'][episode_id])

    saved_features = load_text_features(instruction_id)
    print("presaved bert shape tokens",saved_features['tokens'].shape)
    print("presaved bert shape features",saved_features['features'].shape)

    if False:
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(
            "https://kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/multi-cased-preprocess/versions/3")
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(
            "https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/multi-cased-l-12-h-768-a-12/versions/4",
            trainable=True)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]      # [batch_size, 768].
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

        embedding_model = tf.keras.Model(text_input, pooled_output)
        output_features = embedding_model(sentences)
        import ipdb; ipdb.set_trace()


    if True:
        preprocess_model = make_bert_preprocess_model()
        encoder = hub.KerasLayer('/home/saumyas/Projects/VLN-CE-Plan/data/bert_models/bert_multi_cased_L-12_H-768_A-12_4/',trainable=True)
        preprocessed_input = preprocess_model(sentences)
        output = encoder(preprocessed_input)
        token_mask = preprocessed_input['input_mask']==1
        output_tokens = output['sequence_output'][token_mask]

        print("new bert shape tokens",np.sum(token_mask))
        print("new bert shape features",output_tokens.shape)

        diff = np.max(saved_features['features'] - output_tokens)
        import ipdb; ipdb.set_trace()
    
    
    if False:
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer('/home/saumyas/Projects/VLN-CE-Plan/data/bert_models/bert_multi_cased_preprocess_3/')
        encoder_inputs = preprocessor(text_input)

        encoder = hub.KerasLayer('/home/saumyas/Projects/VLN-CE-Plan/data/bert_models/bert_multi_cased_L-12_H-768_A-12_4/',trainable=True)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]      # [batch_size, 768].
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

        embedding_model_pooled = tf.keras.Model(text_input, pooled_output)
        embedding_model_sequence = tf.keras.Model(text_input, sequence_output)

        out_pooled = embedding_model_pooled(sentences)
        out_sequence = embedding_model_sequence(sentences)
        
        print("new bert shape pooled",out_pooled.shape)
        print("new bert shape sequence",out_sequence.shape)
        import ipdb; ipdb.set_trace()