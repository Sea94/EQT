# QuestionInIntentsAndActs

### Introduction

Effective question-asking is a crucial component of a successful conversational chatbot. It could help the bots manifest empathy and render the interaction more engaging by demonstrating attention to the speaker’s emotions. However, current dialog generation approaches do not model this subtle emotion regulation technique due to the lack of a taxonomy of questions and their purpose in social chitchat. To address this gap, we developed an empathetic question taxonomy (EQT), with special attention paid to questions’ ability to capture communicative acts (Question acts) and their emotion regulation intents (Question intents).

## Question Acts

Question acts capture semantic-driven communicative actions of questions. 

1. **Request information** Ask for new factual information.
2. **Ask about consequence** Ask about the result of the described action or situation.
3. **Ask about antecedent** Ask about the reason or cause of the described state or event.
4. **Suggest a solution** Provide a specific solution to a problem in a form of a question. 
5. **Ask for confirmation** Ask a question to confirm or verify the listener’s understanding of something that has been described by the speaker.
6. **Suggest a reason** Suggest a specific reason or cause of the event or state described by the speaker in a form of a question.
7. **Irony** Ask a question that suggests the opposite of what the speaker may expect, usually to be humorous or pass judgement.
8. **Negative rhetoric** Ask a question to express a critical opinion or validate a speaker’s negative point without expecting an answer.
9. **Positive rhetoric** Ask a question to make an encouraging statement or demonstrate agreement with the speaker about a positive point without expecting an answer.

## Question Intents

Question intents describe the emotional effect the question should have on the dialog partner.

1. **Express interest** Express the willingness to learn or hear more about the subject brought up by the speaker; demonstrate curiosity.
2. **Express concern** Express anxiety or worry about the subject brought up by the speaker.
3. **Offer relief** Reassure the speaker who is anxious or distressed.
4. **Sympathize** Express feelings of pity and sorrow for the speaker’s misfortune.
5. **Support** Offer approval, comfort, or encouragement to the speaker, demonstrate an interest in and concern for the speaker’s success.
6. **Amplify pride** Reinforce the speaker’s feeling of pride.
7. **Amplify excitement** Reinforce the speaker’s feeling of excitement.
8. **Amplify joy** Reinforce the speaker’s glad feeling such as pleasure, enjoyment, or happiness.
9. **De-escalate** Calm down the speaker who is agitated, angry, or temporarily out of control.
10. **Pass judgement** Express a (critical) opinion about the subject brought up by the speaker.
11. **Motivate** Encourage the speaker to move onward.
12. **Moralize speaker** Judge the speaker.

This repository contains the code used to train and evaluate the classifier, the datasets used and the annotated results.  

## Initial Weights
Add the following weights files to the `weights` folder:

https://drive.google.com/drive/folders/1zsPHHqs5rsIydBVER-T2wnyAqM0S6DS-?usp=sharing


## Training

Run `python train_qbert.py [type/intent]`


## Prediction

Run `python predict_qbert.py [type/intent]`


## Requirements

numpy==1.19.2  
tensorflow_gpu==2.5.0  
tqdm==4.56.0  
pytorch_transformers==1.2.0  
pandas==1.2.3  
scikit_learn==0.24.2  
tensorflow==2.6.0  


## TODO

Change `type` to `act`
