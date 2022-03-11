import numpy as np
import pandas as pd
import tensorflow as tf

from os import mkdir
from os.path import join, exists
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Utterances weights.
w = 2 ** np.arange(50)

def create_dataset(tokenizer, data, class_type, buffer_size, batch_size, max_length, lab_mapping):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    # Beginning and end of sentence tokens.
    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    df = data.copy()
    
    # Split into train/validation.
    y = df[class_type].copy().to_frame()
    X = df
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1234)

    # No longer create test dataset since we saved it separately.
    # X_test, X_val, y_test, y_val = train_test_split(
    #     X_test, y_test, stratify=y_test, test_size=0.5)
    
    train_ids = X_train['id'].values.tolist()
    val_ids = X_val['id'].values.tolist()
    # test_ids = X_test['id'].values.tolist() 

    def create_dataset(split_ids):
        # Keep only rows whose DialogID is in split_ids.
        df_split = df[df['id'].isin(split_ids)]

        # Get the dialogues as strings in a list.
        dialogs = df_split['utterance_truncated'].tolist()

        # Get the classes (labels) as strings in a list.
        labels = df_split[class_type].tolist()

        # Map the labels to their numerical integer encodings.
        labels = [lab_mapping[label] for label in labels]
        labels = np.array(labels, dtype=np.int32)

        inputs = np.ones((len(dialogs), max_length), dtype=np.int32)
        weights = np.ones((len(dialogs), max_length), dtype=np.float32)

        for i, dialog in tqdm(enumerate(dialogs), total=len(dialogs)):
            # Split each dialogue into utterances based on \n.
            uttrs = dialog.split('\n')

            for j in range(len(uttrs)):
                # If any utterance in the dialogue starts with `- `, remove it.
                if uttrs[j].startswith('- '):
                    uttrs[j] = uttrs[j][2:]
            
            # List storing the token encodings of each utterance.
            uttr_ids = []

            # List storing the overall weight of each utterance.
            weight = []

            # Sum of all utterance weights.
            total_weight = np.sum(w[:len(uttrs)])

            for j in range(len(uttrs)-1, -1, -1):
                # Tokenize current utterance.
                encoded = tokenizer.encode(uttrs[j])

                # Add overall weight of current utterance to the weights list.
                weight += [w[j] / total_weight] * (len(encoded) + 2)

                # Add start of sentence and end of sentence tokens to utterance's token encoding.
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]

            # Keep max. 100 tokens and weights from all the utterances.    
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]

            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID

            # Store utterance encoding in the inputs matrix.
            inputs[i,:len(uttr_ids)] = uttr_ids

            # Store utterance's weights in the weights matrix.
            weights[i,:len(uttr_ids)] = weight

        # Check we have as many dialogues as labels.    
        assert inputs.shape[0] == labels.shape[0]
        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights, labels

    train_inputs, train_weights, train_labels = create_dataset(train_ids)
    val_inputs, val_weights, val_labels = create_dataset(val_ids)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_inputs),
        tf.data.Dataset.from_tensor_slices(train_weights),
        tf.data.Dataset.from_tensor_slices(train_labels))
    
    val_dataset = (tf.data.Dataset.from_tensor_slices(val_inputs),
        tf.data.Dataset.from_tensor_slices(val_weights),
        tf.data.Dataset.from_tensor_slices(val_labels))

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    val_dataset = tf.data.Dataset.zip(val_dataset).batch(batch_size)

    return train_dataset, val_dataset #, test_ids


def create_test_dataset(tokenizer, data, class_type, batch_size, max_length, lab_mapping):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))
    
    # Beginning and end of sentence tokens.
    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    df = data.copy()            

    def create_dataset(split_ids):
        df_split = df[df['id'].isin(split_ids)]
        
        # Get the classes (labels) as strings in a list.
        labels = df_split[class_type].tolist()
    
        # Map the labels to their numerical integer encodings.
        labels = [lab_mapping[label] for label in labels]
        labels = np.array(labels, dtype=np.int32)

        # Get the dialogues as strings in a list.
        dialogs = df_split['utterance_truncated'].tolist()
    
        inputs = np.ones((len(dialogs), max_length), dtype=np.int32)
        weights = np.ones((len(dialogs), max_length), dtype=np.float32)
        
        for i, dialog in tqdm(enumerate(dialogs), total=len(dialogs)):
            # Split each dialogue into utterances based on \n.
            uttrs = dialog.split('\n')
            
            for j in range(len(uttrs)):
                # If any utterance in the dialogue starts with `- `, remove it.
                if uttrs[j].startswith('- '):
                    uttrs[j] = uttrs[j][2:]
                    
            # List storing the token encodings of each utterance.
            uttr_ids = []

            # List storing the overall weight of each utterance.
            weight = []
            
            # Sum of all utterance weights.
            total_weight = np.sum(w[:len(uttrs)])
            
            for j in range(len(uttrs)-1, -1, -1):
                # Tokenize current utterance.
                encoded = tokenizer.encode(uttrs[j])
                
                # Add overall weight of current utterance to the weights list.
                weight += [w[j] / total_weight] * (len(encoded) + 2)

                # Add start of sentence and end of sentence tokens to utterance's token encoding.
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                
            # Keep max. 100 tokens and weights from all the utterances. 
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]
            
            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID
            
            # Store utterance encoding in the inputs matrix.
            inputs[i,:len(uttr_ids)] = uttr_ids

            # Store utterance's weights in the weights matrix.
            weights[i,:len(uttr_ids)] = weight

        # Check we have as many dialogues as labels. 
        assert inputs.shape[0] == labels.shape[0]
        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights, labels

    test_ids = df['id'].values.tolist()
    test_inputs, test_weights, test_labels = create_dataset(test_ids)
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_inputs),
        tf.data.Dataset.from_tensor_slices(test_weights),
        tf.data.Dataset.from_tensor_slices(test_labels))
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return test_dataset


def create_test_dataset_for_prediction(tokenizer, data, batch_size, max_length, lab_mapping):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))
    
    # Beginning and end of sentence tokens.
    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    df = data.copy()
            
    def create_dataset():
        ids = df['id'].tolist()

        # Get the dialogues as strings in a list.
        dialogs = df['utterance_truncated'].tolist()
    
        inputs = np.ones((len(dialogs), max_length), dtype = np.int32)
        weights = np.ones((len(dialogs), max_length), dtype = np.float32)
        
        for i, dialog in tqdm(enumerate(dialogs), total = len(dialogs)):
            # Split each dialogue into utterances based on \n.
            uttrs = dialog.split('\n')
            
            for j in range(len(uttrs)):
                # If any utterance in the dialogue starts with `- `, remove it.
                if uttrs[j].startswith('- '):
                    uttrs[j] = uttrs[j][2:]
                    
            # List storing the token encodings of each utterance.
            uttr_ids = []

            # List storing the overall weight of each utterance.
            weight = []
            
            # Sum of all utterance weights.
            total_weight = np.sum(w[:len(uttrs)])
            
            for j in range(len(uttrs)-1, -1, -1):
                # Tokenize current utterance.
                encoded = tokenizer.encode(uttrs[j])
                
                # Add overall weight of current utterance to the weights list.
                weight += [w[j] / total_weight] * (len(encoded) + 2)

                # Add start of sentence and end of sentence tokens to utterance's token encoding.
                uttr_ids += [EOS_ID] + encoded + [EOS_ID]
                
            # Keep max. 100 tokens and weights from all the utterances. 
            uttr_ids = uttr_ids[:max_length]
            weight = weight[:max_length]
            
            # Is this really necessary? The SOS especially?
            uttr_ids[0] = SOS_ID
            uttr_ids[-1] = EOS_ID
            
            # Store utterance encoding in the inputs matrix.
            inputs[i,:len(uttr_ids)] = uttr_ids

            # Store utterance's weights in the weights matrix.
            weights[i,:len(uttr_ids)] = weight

        # Check we have as many dialogues as labels. 
        print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, weights, ids

    test_inputs, test_weights, test_ids = create_dataset()

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_inputs),
                    tf.data.Dataset.from_tensor_slices(test_weights),
                    tf.data.Dataset.from_tensor_slices(test_ids))

    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return test_dataset
