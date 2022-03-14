The `ed_annotated.csv` file contains a dataframe with question act and question intent labels
assigned to each listener's question in the EmpatheticDialogues (ED) dataset. Each row represents
one dialog turn, following the original format by Rashkin et al. [1]. The columns in the dataframe
are defined as follows:
- `conv_id` (str): unique id of each dialog
- `utterance_idx` (int): consecutive number of the given turn in the dialog
- `context` (str): emotion label given to the speaker to come up with a grounding situation
- `prompt` (str): situation described by the speaker
- `speaker_idx` (int): unique id of the given speaker. One speaker can take part in several
	dialogs.
- `utterance` (str): a dialog turn produced by the given speaker for the given dialog
- `set_id` (str): identification of the dataset to which the dialog belongs according to the
	split given in [1] (train, valid, or test)
- `utterance_split` (list of str): speaker's utterance split into individual sentences
- `question_count` (int): number of questions in the given utterance. Takes values equal to or
	above 0 for listeners' utterances (with even `utterance_idx`) and -1 for speakers' utterances
- `labeled_question_idx` (empty str or list of int): for utterances with `question_count` above 0
	provides the indices of questions in `utterance_split`
- `act_list` (list of str): for utterances with non-empty `labeled_question_idx` provides a list
	of act labels assigned to each question in the utterance identified by `labeled_question_idx`
- `intent_list` (list of str): for utterances with non-empty `labeled_question_idx` provides a list
	of intent labels assigned to each question in the utterance identified by `labeled_question_idx`
- `act_source` and `intent_source` (list of str): for utterances with non-empty `labeled_question_idx`
	provides a list of sources from which the corresponding labels from the `act_list` originated 
	(manual, mturk, SBERT, QBERT).



[1] H. Rashkin, E. M. Smith, M. Li, Y. Boureau Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset
