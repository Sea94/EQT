import pandas as pd
import pickle


def preprocess_data(data_path, class_type, test_set=None, drop_test=True):
	print('Reading data from pickle file...')
	with open(data_path, 'rb') as f:
	    quest_df = pickle.load(f)

	quest_df_types = quest_df[(quest_df[class_type] != '')].copy()
	quest_df_types['utterance_truncated'] = quest_df_types['utterance_truncated'].apply(lambda x: ' '.join(x))

	quest_df_types['utterance_truncated'] = quest_df_types[['id','utterance_truncated','type', 'intent']].groupby(['id'])['utterance_truncated'].transform(lambda x: ' \n '.join(x))
	data = quest_df_types[['id', 'utterance_truncated','type', 'intent']].drop_duplicates()


	if drop_test and test_set is None:
		all_labelled = data[(data['type'] != '') & (data['intent'] != '')].copy()
		test_set = all_labelled.sample(n=1500, random_state=1234)

		print('Dumping test set to pickle file...')
		with open('./data/test_set.pickle', 'wb') as f:
			pickle.dump(test_set, f)
	
	if drop_test:
		data = data.drop(test_set.index)

	return data
