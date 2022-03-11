import argparse
import time
import numpy as np
import pickle
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from datasets import create_test_dataset_for_prediction
from optimize import CustomSchedule
from model_utils import *
from model_qbert import QBERT, loss_function
from preprocess_data import preprocess_data


# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
# num_emotions = 9 # 41  # Number of emotion categories

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size


adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

mappings = {
    'type': {
        'Ask about antecedent': 0,
        'Ask about consequence': 1,
        'Ask for confirmation': 2,
        'Irony': 3,
        'Negative rhetoric': 4,
        'Positive rhetoric': 5,
        'Request information': 6,
        'Suggest a reason': 7,
        'Suggest a solution': 8
    },
    'intent': {
        'Amplify excitement': 0,
        'Amplify joy': 1,
        'Amplify pride': 2,
        'De-escalate': 3,
        'Express concern': 4,
        'Express interest': 5,
        'Moralize speaker': 6,
        'Motivate': 7,
        'Offer relief': 8,
        'Pass judgement': 9,
        'Support': 10,
        'Sympathize': 11
    }
}


def get_label_source(row, class_tag):
  if row[class_tag] != '':
    return row['label_source']
  return 'QBERT'


def main():
    # Initialize arguments parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('class_type', help='Specify whether the model is trained for question acts or intents.')
    parser.add_argument('-e', '--epoch', type=int, help='Specify epoch of parameters to restore.', default=4)
    parser.add_argument('-d', '--data_path', help='Specify unlabelled data path.', default='./data/quest_df_all_labelled_intents.pickle')
    parser.add_argument('-l', '--labelled_path', help='Specify path to save labelled data at.', default='./results/data_labelled.pickle')
    parser.add_argument('-bs', '--batch_size', type=int, help='Specify batch size.', default=5)
    parser.add_argument('-ml', '--max_len', type=int, help='Specify max. number of tokens.', default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Specify learning rate.', default=2e-5)
    parser.add_argument('-cp', '--checkpoints_path', help='Specify checkpoints path.')
    args = parser.parse_args()

    class_type = args.class_type
    data_path = args.data_path
    labelled_path = args.labelled_path
    restore_epoch = args.epoch
    max_length = args.max_len
    batch_size = args.batch_size
    peak_lr = args.learning_rate

    if not args.checkpoints_path:
        checkpoints_path = './checkpoints/qbert_high_sim_weighted/{}s'.format(class_type)

    lab_mapping = mappings[class_type]

    pred_mapping = {v: k for k, v in lab_mapping.items()}

    num_classes = len(pred_mapping.keys())

    with open(data_path, 'rb') as f:
        quest_df = pickle.load(f)

    quest_df['{}_source'.format(class_type)] = ''
    quest_df['{}_source'.format(class_type)] = quest_df.apply(get_label_source, axis=1, class_tag=class_type)

    data = preprocess_data(data_path, class_type, test_set=None, drop_test=False)

    test_dataset = create_test_dataset_for_prediction(tokenizer, data, batch_size, max_length, lab_mapping)


    # Define the model.
    qbert = QBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_classes)

    # Define optimizer and metrics.
    learning_rate = peak_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2,
        epsilon = adam_epsilon)

    def predict(model_name):
      y_pred = []
      pred_ids = []
      for inputs in test_dataset:
          inp, weights, ids = inputs
          enc_padding_mask = create_masks(inp)
          pred_class = qbert(inp, weights, False, enc_padding_mask)
          pred_class = np.argmax(pred_class.numpy(), axis=1)
          y_pred += pred_class.tolist()
          y_pred_lab = [pred_mapping[pred] for pred in y_pred]
          pred_labels = np.array(y_pred_lab)
          pred_ids += [idx.decode('utf-8') for idx in ids.numpy()]

      return pred_labels, pred_ids

    def validate(model_name):
        y_true = []
        y_pred = []
        for inputs in test_dataset:
            inp, weights, tar_class = inputs
            enc_padding_mask = create_masks(inp)
            pred_class = qbert(inp, weights, False, enc_padding_mask)
            pred_class = np.argmax(pred_class.numpy(), axis=1)
            y_true += tar_class.numpy().tolist()
            y_pred += pred_class.tolist()

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print('All -- P: {:.4f}, R: {:.4f}, F: {:.4f}, A: {:.4f}\n'.format(p, r, f, acc))


    def get_individual_scores(model_name):
        y_true = []
        y_pred = []
        for inputs in test_dataset:
            inp, weights, tar_class = inputs
            enc_padding_mask = create_masks(inp)
            pred_class = qbert(inp, weights, False, enc_padding_mask)
            pred_class = np.argmax(pred_class.numpy(), axis=1)
            y_true += tar_class.numpy().tolist()
            y_pred += pred_class.tolist()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=list(range(num_classes)))
        with open('./metrics_types/individual_scores_{}.csv'.format(model_name), 'w') as f_out:
            f_out.write('class,precision,recall,f_score\n')
            for i in range(num_classes):
                f_out.write('{},{:.4f},{:.4f},{:.4f}\n'.format(pred_mapping[i], p[i], r[i], f[i]))

    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model=qbert, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoints_path, max_to_keep=None)
    ckpt.restore(ckpt_manager.checkpoints[restore_epoch - 1]).expect_partial()
    print('Checkpoint {} restored!!!'.format(ckpt_manager.checkpoints[restore_epoch - 1]))
    # validate('extra_3k_weighted') 
    predictions, ids = predict('extra_3k_weighted')

    # Label the data points.
    for idx, pred in zip(ids, predictions):
        quest_df.loc[quest_df['id'] == idx, class_type] = pred

    with open(labelled_path, 'wb') as f:
        pickle.dump(quest_df, f)


if __name__ == '__main__':
    main()
