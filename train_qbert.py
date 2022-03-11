import argparse
import csv
import sys
import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
import pickle
from pytorch_transformers import RobertaTokenizer

from optimize import CustomSchedule
from model_utils import *
from model_qbert import QBERT, loss_function
from preprocess_data import preprocess_data
from sklearn import preprocessing
from datasets import create_dataset, create_test_dataset


# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
# num_emotions = 12  # Number of class categories, was 41 originally.

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size

# max_length = 100  # Maximum number of tokens
# buffer_size = 100000
# batch_size = 256
# batch_size = 5
# num_epochs = 15
# peak_lr = 2e-5
# warmup_steps = 44
# total_steps = 440
adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6


def main():

    # Initialize arguments parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('class_type', help='Specify whether the model is trained for question acts or intents.')
    parser.add_argument('-d', '--data_path', help='Specify data path.', default='./data/quest_df_all_with_sbert.pickle')
    parser.add_argument('-td', '--test_data_path', help='Specify test data path.', default='./data/test_set.pickle')
    parser.add_argument('-bs', '--batch_size', type=int, help='Specify batch size.', default=5)
    parser.add_argument('-bfs', '--buffer_size', type=int, help='Specify buffer size.', default=100000)
    parser.add_argument('-ml', '--max_len', type=int, help='Specify max. number of tokens.', default=100)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs.', default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Specify learning rate.', default=2e-5)
    parser.add_argument('-w', '--weights_path', help='Specify intial weights path.')
    parser.add_argument('-cp', '--checkpoints_path', help='Specify checkpoints path.')
    parser.add_argument('-l', '--log_path', help='Specify logs path.')
    args = parser.parse_args()

    class_type = args.class_type
    data_path = args.data_path
    test_data_path = args.test_data_path
    max_length = args.max_len
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    num_epochs = args.epochs
    peak_lr = args.learning_rate

    if not args.checkpoints_path:
        checkpoints_path = './checkpoints/qbert_high_sim_weighted/{}s'.format(class_type)

    if not args.log_path:
        log_path = './log/{}s/qbert_high_sim_weighted.log'.format(class_type)

    if not args.weights_path:
        weights_path = './weights/roberta2qbert_{}s.h5'.format(class_type)

    print('Reading test set from pickle file...')
    with open(test_data_path, 'rb') as f:
        test_set = pickle.load(f)

    data = preprocess_data(data_path, class_type, test_set=test_set)

    labels = list(data[class_type].unique())
    num_classes = len(labels)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    transformed = le.transform(labels)
    lab_mapping = dict(zip(labels, transformed))

    if not exists('log'):
        mkdir('log')

    f = open(log_path, 'a', encoding = 'utf-8')

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():

        train_dataset, val_dataset = create_dataset(tokenizer, data, class_type, buffer_size, batch_size, max_length, lab_mapping)
        test_dataset = create_test_dataset(tokenizer, test_set, class_type, batch_size, max_length, lab_mapping)
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        
        # Define the model.
        qbert = QBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, max_position_embed, vocab_size, num_classes)
        
        # Build the model and initialize weights from PlainTransformer pre-trained on OpenSubtitles.
        build_model(qbert, max_length, vocab_size)
        qbert.load_weights(weights_path)

        print('Weights initialized from RoBERTa.')
        f.write('Weights initialized from RoBERTa.\n')
        

        # Define optimizer and metrics.
        # learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
        learning_rate = peak_lr
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2,
            epsilon = adam_epsilon)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        
        # Define the checkpoint manager.
        ckpt = tf.train.Checkpoint(model=qbert, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoints_path, max_to_keep=None)
        
        # If a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            f.write('Latest checkpoint restored!!\n')
            
        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                inp, weights, tar_class = inputs
                enc_padding_mask = create_masks(inp)

                with tf.GradientTape() as tape:
                    pred_class = qbert(inp, weights, True, enc_padding_mask)  # (batch_size, num_emotions)
                    losses_per_examples = loss_function(tar_class, pred_class)
                    loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                gradients = tape.gradient(loss, qbert.trainable_variables)
                optimizer.apply_gradients(zip(gradients, qbert.trainable_variables))
                return loss

            losses_per_replica = mirrored_strategy.run(step_fn, args = (dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)

            train_loss(mean_loss)
            return mean_loss
        
        
        @tf.function
        def valid_step(dist_inputs, my_loss):
            def step_fn(inputs):
                inp, weights, tar_class = inputs
                enc_padding_mask = create_masks(inp)

                pred_class = qbert(inp, weights, False, enc_padding_mask)  # (batch_size, num_emotions)
                losses_per_examples = loss_function(tar_class, pred_class)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

                return loss

            losses_per_replica = mirrored_strategy.run(step_fn, args=(dist_inputs,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)
            my_loss(mean_loss)
            
            
        def validate(dataset):
            accuracy = []
            for (batch, inputs) in enumerate(dataset):
                inp, weights, tar_class = inputs
                enc_padding_mask = create_masks(inp)
                pred_class = qbert(inp, weights, False, enc_padding_mask)
                pred_class = np.argmax(pred_class.numpy(), axis=1)
                accuracy += (tar_class.numpy() == pred_class).tolist()
            return np.mean(accuracy)

        # Start training
        for epoch in range(num_epochs):

            if epoch == 0:
                # RUN ONLY BEFOR EPOCH 1.
                with open('./metrics/{}s/train_results_{}s.csv'.format(class_type, class_type), mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])

            start = time.time()

            train_loss.reset_states()

            for (batch, inputs) in enumerate(train_dataset):
                current_loss = train_step(inputs)
                current_mean_loss = train_loss.result()
                print('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}'.format(
                    epoch + 1, batch, current_mean_loss, current_loss))
                f.write('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                    epoch + 1, batch, current_mean_loss, current_loss))

            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            f.write('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

            epoch_loss = train_loss.result()
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, epoch_loss))
            f.write('Epoch {} Loss {:.4f}\n'.format(epoch + 1, epoch_loss))

            train_ac = validate(train_dataset)
            print('Epoch {} Train accuracy {:.4f}'.format(epoch + 1, train_ac))
            f.write('Epoch {} Train accuracy {:.4f}\n'.format(epoch + 1, train_ac))

            current_time = time.time()
            print('Time taken for 1 epoch: {} secs'.format(current_time - start))
            f.write('Time taken for 1 epoch: {} secs\n'.format(current_time - start))

            val_loss.reset_states()
            for inputs in val_dataset:
                valid_step(inputs, val_loss)
            epoch_val_loss = val_loss.result()
            print('Epoch {} Validation loss {:.4f}'.format(epoch + 1, epoch_val_loss))
            f.write('Epoch {} Validation loss {:.4f}\n'.format(epoch + 1, epoch_val_loss))

            val_ac = validate(val_dataset)
            print('Epoch {} Validation accuracy {:.4f}'.format(epoch + 1, val_ac))
            f.write('Epoch {} Validation accuracy {:.4f}\n'.format(epoch + 1, val_ac))

            test_loss.reset_states()
            for inputs in test_dataset:
                valid_step(inputs, test_loss)
            test_val_loss = test_loss.result()
            print('Epoch {} Test loss {:.4f}'.format(epoch + 1, test_val_loss))
            f.write('Epoch {} Test loss {:.4f}\n'.format(epoch + 1, test_val_loss))

            test_ac = validate(test_dataset)
            print('Epoch {} Test accuracy {:.4f}\n'.format(epoch + 1, test_ac))
            f.write('Epoch {} Test accuracy {:.4f}\n\n'.format(epoch + 1, test_ac))

            with open('./metrics/{}s/train_results_{}s.csv'.format(class_type, class_type), mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([str(epoch_loss.numpy()), str(train_ac), str(epoch_val_loss.numpy()), str(val_ac), str(test_val_loss.numpy()), str(test_ac)])

    f.close()

if __name__ == '__main__':
    main()
