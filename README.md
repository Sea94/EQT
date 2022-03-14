# EQT: Empathetic Question Taxonomy

### Introduction

Effective question-asking is a crucial component of a successful conversational chatbot. It could help the bots manifest empathy and render the interaction more engaging by demonstrating attention to the speaker’s emotions. However, current dialog generation approaches do not model this subtle emotion regulation technique due to the lack of a taxonomy of questions and their purpose in social chitchat. To address this gap, we developed an empathetic question taxonomy (EQT), with special attention paid to questions’ ability to capture communicative acts (Question acts) and their emotion regulation intents (Question intents).

#### Question Acts

Question acts capture semantic-driven communicative actions of questions. 

1. **Request information**: Ask for new factual information.
2. **Ask about consequence**: Ask about the result of the described action or situation.
3. **Ask about antecedent**: Ask about the reason or cause of the described state or event.
4. **Suggest a solution**: Provide a specific solution to a problem in a form of a question. 
5. **Ask for confirmation**: Ask a question to confirm or verify the listener’s understanding of something that has been described by the speaker.
6. **Suggest a reason**: Suggest a specific reason or cause of the event or state described by the speaker in a form of a question.
7. **Irony**: Ask a question that suggests the opposite of what the speaker may expect, usually to be humorous or pass judgement.
8. **Negative rhetoric**: Ask a question to express a critical opinion or validate a speaker’s negative point without expecting an answer.
9. **Positive rhetoric**: Ask a question to make an encouraging statement or demonstrate agreement with the speaker about a positive point without expecting an answer.

#### Question Intents

Question intents describe the emotional effect the question should have on the dialog partner.

1. **Express interest**: Express the willingness to learn or hear more about the subject brought up by the speaker; demonstrate curiosity.
2. **Express concern**: Express anxiety or worry about the subject brought up by the speaker.
3. **Offer relief**: Reassure the speaker who is anxious or distressed.
4. **Sympathize**: Express feelings of pity and sorrow for the speaker’s misfortune.
5. **Support**: Offer approval, comfort, or encouragement to the speaker, demonstrate an interest in and concern for the speaker’s success.
6. **Amplify pride**: Reinforce the speaker’s feeling of pride.
7. **Amplify excitement**: Reinforce the speaker’s feeling of excitement.
8. **Amplify joy**: Reinforce the speaker’s glad feeling such as pleasure, enjoyment, or happiness.
9. **De-escalate**: Calm down the speaker who is agitated, angry, or temporarily out of control.
10. **Pass judgement**: Express a (critical) opinion about the subject brought up by the speaker.
11. **Motivate**: Encourage the speaker to move onward.
12. **Moralize speaker**: Judge the speaker.

We used an iterative qualitative coding method to manually annotate more than 310 listener questions from the EmpatheticDialogues dataset (Rashkin et al., 2019) using the above taxonomy. A larger sub-sample of the EmpatheticDialogues dataset was annotated recruiting crowdworkers from Amazon Mechanical Turk (MTurk) resulting in 6,433 questions assigned with a question act label and 5,826 questions assigned with a question intent label. 

We extended the number of examples for each question act and intent by searching through the rest of the dataset using k-Nearest-Neighbors (k-NN) method. More specifically, we employed the Sentence-BERT (SBERT) framework (Reimers and Gurevych, 2019) to obtain embeddings for all questions with their contexts. Then we used the cosine similarity measure to find k labeled NNs for each question in the unlabeled set and assign the same labels to them. This produced additional 1,911 labels for question acts and 1,886 labels for question intents.

Using the human-annotated and augmented labels, we trained two classifiers, which we collectively call QBERT.

This repository contains the code used to train and evaluate the QBERT classifier, the datasets used for training and evaluation and the annotated results. 

### QBERT

QBERT models for predicting question acts and intents have identical architecture and vary only in the number of output categories in the final layer. Each model consists of a BERT-based representation network, an attention layer, one hidden layer, and a softmax layer. For the representation network, we used the architecture with 12 layers, 768 dimensions, 12 heads,and 110M parameters. We initialized it with the weights of RoBERTa language model pre-trained by Liu et al. (2019) and for training used the same hyper-parameters as the authors.

The listener question and preceding dialog turns are fed into the classfier in the reverse order to obtain predictions. 

#### Initial Weights
Add the following weights files to the `weights` folder:

https://drive.google.com/drive/folders/1zsPHHqs5rsIydBVER-T2wnyAqM0S6DS-?usp=sharing


#### Training

Run `python train_qbert.py [type/intent]`


#### Prediction

Run `python predict_qbert.py [type/intent]`


#### Dependencies

numpy==1.19.2  
tensorflow_gpu==2.5.0  
tqdm==4.56.0  
pytorch_transformers==1.2.0  
pandas==1.2.3  
scikit_learn==0.24.2  
tensorflow==2.6.0  

### Datasets

The original EmpatheticDialogues dataset can be downloaded from: [github.com/facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues).

This repository contains the following datasets:

1. ***data/quest_df_all_labelled_intents.pickle***: To do
2. ***data/quest_df_all_with_sbert.pickle***: Training data required for training QBERT
3. ***data/test_set.pickle***: Testing data used for testing the performance of QBERT
4. **annotations/ed_annotated.csv**: Contains question act and question intent labels assigned to each listener's question in the EmpatheticDialogues (ED) dataset as predicted by QBERT.

### References

Please cite the following paper if you found the resources in this repository useful.

- Ekaterina Svikhnushina, Iuliana Voinea, Anuradha Welivita, and Pearl Pu **A Taxonomy of Empathetic Questions in Social Dialogs**, In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)*.  

### Bibliography

- Hannah Rashkin, Eric Michael Smith, Margaret Li and Y-Lan Boureau. 2019.  Towards Empathetic Open-domain Conversation  Models:  A  New  Benchmark  and  Dataset.   In *Proceedings  of  the  57th  Annual  Meeting  of  the Association for Computational Linguistics*, pages 5370–5381, Florence, Italy.
- Nils Reimers and Iryna Gurevych. 2019. Sentence786 BERT: Sentence embeddings using Siamese BERT networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3982–3992.
- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*.
