import torch

from treelstm import TreeLSTM, calculate_evaluation_orders, batch_tree_input, TreeDataset, convert_tree_to_tensors

from torch.utils.data import Dataset, IterableDataset, DataLoader

import os
import codecs
from sklearn.metrics import f1_score
import random
import numpy as np

seed_val = 12

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


tree_path = 'C:\\Users\\Harshavardhan\\IR-term-project\\paper-1\\GITHUB_REPO\\Parsed-Trees\\'
#tree_path = 'C:\\Users\\Harshavardhan\\IR-term-project\\paper-1\\GITHUB_REPO\\Parsed-Trees_3\\'
log_file = "log.txt"

test_set = 	[
			'charliehebdo-all-rnr-threads.txt',
			'ebola-essien-all-rnr-threads.txt', 
			'germanwings-crash-all-rnr-threads.txt', 
			'sydneysiege-all-rnr-threads.txt', 
			'gurlitt-all-rnr-threads.txt', 
			'prince-toronto-all-rnr-threads.txt', 
			'ottawashooting-all-rnr-threads.txt', 
			'putinmissing-all-rnr-threads.txt', 
			'ferguson-all-rnr-threads.txt'
			]

from random import shuffle

IN_FEATURES = 40
OUT_FEATURES = 2
NUM_ITERATIONS = 100
BATCH_SIZE = 50
HIDDEN_UNITS = 128
HIDDEN_STANCE_UNITS = 128
OUT_STANCE_FEATURES = 4
DROP_OUT = 0.1
LEARNING_RATE = 0.001
no_stance = 0
ENTIRE_DATASET_WEIGHTS=True
#weights = [0,0.1192,1.0,0.7981,0.511]
#weights = [0.123,1.4,0.721,0.671]
weights = [0.0436,0.4141,0.3305,0.2118]
#weights = [0.128,1.3,0.848,0.595]

files = os.listdir(tree_path)


for test_file in test_set:
	#logging
	f = open(log_file,'a')
	print('Training Set:', set(files) - {test_file})
	f.write('Training Set:{files}\n'.format(files = set(files) - {test_file}))
	f.close()

	test_trees = []
	train_trees = []

	for filename in files:
		input_file = codecs.open(tree_path + filename, 'r', 'utf-8')

		tree_li = []
		pos_trees = []
		neg_trees = []

		for row in input_file:
			s = row.strip().split('\t')

			tweet_id = s[0]
			curr_tree = eval(s[1])
			# print(tweet_id)
			try:
				curr_tensor, curr_label = convert_tree_to_tensors(curr_tree)
			except Exception as e:
				continue

			curr_tensor['tweet_id'] = tweet_id

			if curr_label == 1:
				pos_trees.append(curr_tensor)
			else:
				neg_trees.append(curr_tensor)

		input_file.close()


		if filename == test_file:
			tree_li = pos_trees + neg_trees
			test_trees = tree_li
	
		else:			
			tree_li = pos_trees + neg_trees

			shuffle(tree_li)

			train_trees += tree_li
	
	weight_0 = 0
	weight_1 = 0
	weight_2 = 0
	weight_3 = 0

	for curr_tree in train_trees:
		pre_stance = (curr_tree['stance']-1).type(torch.long)
		for val in pre_stance:
			if val == 0:
				weight_0 += 1
			if val == 1:
				weight_1 += 1
			if val == 2:
				weight_2 += 1
			if val == 3:
				weight_3 += 1
	if(ENTIRE_DATASET_WEIGHTS):
		for curr_tree in test_trees:
			pre_stance = (curr_tree['stance']-1).type(torch.long)
			for val in pre_stance:
				if val == 0:
					weight_0 += 1
				if val == 1:
					weight_1 += 1
				if val == 2:
					weight_2 += 1
				if val == 3:
					weight_3 += 1
	
	print(weight_0,weight_1,weight_2,weight_3)
	k=1 / (1/weight_0 + 1/weight_1 + 1/weight_2 + 1/weight_3)
	weights = [k/weight_0,k/weight_1,k/weight_2,k/weight_3]
	print(weights)
	#logging
	f = open(log_file,'a')
	f.write('count:{w1},{w2},{w3},{w4}\nweights:{array}\n'.format(w1=weight_0,w2=weight_1,w3=weight_2,w4=weight_3,array=weights))
	f.close()
	model = TreeLSTM(IN_FEATURES, OUT_FEATURES, HIDDEN_UNITS, HIDDEN_STANCE_UNITS, OUT_STANCE_FEATURES, DROP_OUT).train()

	loss_function = torch.nn.CrossEntropyLoss()
	# loss_binary = torch.nn.BCELoss()
	# loss_binary = torch.nn.BCEWithLogitsLoss()
	# loss_binary = torch.nn.CrossEntropyLoss(ignore_index=-1)
	class_weights = torch.FloatTensor(weights)
	loss_binary = torch.nn.CrossEntropyLoss(ignore_index=no_stance-1,weight=class_weights)

	optimizer = torch.optim.Adam(model.parameters() , lr = LEARNING_RATE)

	for i in range(NUM_ITERATIONS):
		total_loss = 0
		total_stance_loss = 0

		optimizer.zero_grad()

		curr_tree_dataset = TreeDataset(train_trees)

		train_data_generator = DataLoader(
			curr_tree_dataset,
			collate_fn=batch_tree_input,
			batch_size=BATCH_SIZE,
			shuffle = True
		)

		for tree_batch in train_data_generator:
			try:
				h, h_root, c = model(
					tree_batch['f'],
					tree_batch['node_order'],
					tree_batch['adjacency_list'],
					tree_batch['edge_order'],
					tree_batch['root_node'],
					tree_batch['root_label']
					)
			except:
				continue
			
			labels = tree_batch['l']
			stance_labels = (tree_batch['stance']).type(torch.long)
			root_labels = tree_batch['root_label']
			# print(h.shape)
			# print(stance_labels.shape)
			# print(h_root.shape,root_labels.shape)

			loss = loss_function(h_root, root_labels)
			loss_stance = loss_binary(h,stance_labels-1)
			#loss_stance = loss_binary(h,stance_labels)
			tot_loss = loss + loss_stance
			tot_loss.backward()

			optimizer.step()

			total_loss += tot_loss
			total_stance_loss += loss_stance

		print(f'Iteration {i+1} Loss: {total_loss}')
		#logging
		f = open(log_file,'a')
		f.write('Iteration {epoch} Loss: {total_loss}\n'.format(epoch=i+1, total_loss=total_loss))
		f.close()
		# print(f'Iteration {i+1} Stance Loss: {total_stance_loss}')
		# logging
		# f = open(log_file,'a')
		# f.write('Iteration {epoch} Stance Loss: {total_stance_loss}\n'.format(epoch=i+1, total_stance_loss = total_stance_loss))
		# f.close()
	
	print('Training Complete')
	#logging
	f = open(log_file,'a')
	f.write('Training Complete\n')
	f.close()

	print('Now Testing:', test_file)
	#logging
	f = open(log_file,'a')
	f.write('Now Testing:{test_file}\n'.format(test_file=test_file))
	f.close()

	acc = 0
	total = 0
	acc_stance = 0
	total_stance = 0
	num_1 = 0
	num_2 = 0
	num_3 = 0
	num_4 = 0

	pred_label_li = []
	true_label_li = []
	pred_stance_label_li = []
	true_stance_label_li = []

	for test in test_trees:
		try:
			h_test, h_test_root, c = model(
				test['f'],
				test['node_order'],
				test['adjacency_list'],
				test['edge_order'],
				test['root_n'],
				test['root_l']
			)
		except:
			continue

		pred_v, pred_label = torch.max(h_test_root, 1)
		pred_stance_vec, pred_stance_label = torch.max(h_test, 1)

		true_label = test['root_l']
		true_stance_label = (test['stance']-1).type(torch.long)

		if pred_label == true_label:
			acc += 1
		# acc_stance += (pred_stance_label == true_stance_label).sum()
		# total_stance += (true_stance_label == true_stance_label).sum()

		for a,b in zip(true_stance_label,pred_stance_label):
			if a == no_stance-1 :
				continue
			if a == b:
				acc_stance +=1
				if b == 0:
					num_1+=1
				if b == 1:
					num_2+=1
				if b == 2:
					num_3+=1
				if b == 3:
					num_4+=1
			total_stance += 1
			pred_stance_label_li.append(b)
			true_stance_label_li.append(a)


		pred_label_li.append(pred_label)
		true_label_li.append(true_label)

		total += 1

	macro_f1 = f1_score(pred_label_li, true_label_li, average = 'macro')
	macro_stance_f1 = f1_score(pred_stance_label_li, true_stance_label_li, average = 'macro')

	print("Report for Rumour classification:-")
	print(test_file, 'accuracy:', acc / total)
	print(test_file, 'macro_f1:', macro_f1)
	print(test_file, 'total tested:', total)
	print("Report for Stance classification:-")
	print("correctly classified Stance labels:",acc_stance,"\nTotal available Stance labels:",total_stance)
	print(num_1,num_2,num_3,num_4)
	if total_stance != 0:
		print(test_file, 'accuracy_stance:', acc_stance / total_stance)
		print(test_file, 'macro_stance_f1:', macro_stance_f1)

	#logging
	f = open(log_file,'a')
	f.write('\nReport for Rumour classification:-\n')
	f.write('{test_file} accuracy:{accuracy}\n'.format(test_file=test_file, accuracy=acc / total))
	f.write('{test_file} macro_f1:{macro_f1}\n'.format(test_file=test_file, macro_f1=macro_f1))
	f.write('{test_file} total tested:{total}\n'.format(test_file=test_file,total=total))
	f.write('\nReport for Stance classification:-\n')
	f.write('correctly classified Stance labels:{acc_stance}\n'.format(acc_stance=acc_stance))
	f.write('Total available Stance labels:{total_stance}\n'.format(total_stance=total_stance))
	f.write('correctly classified Stance labels for each class:{num_1},{num_2},{num_3},{num_4}\n'.format(num_1=num_1,num_2=num_2,num_3=num_3,num_4=num_4))
	if total_stance != 0:
		f.write('{test_file} accuracy_stance:{accuracy_stance}\n'.format(test_file=test_file, accuracy_stance=acc_stance / total_stance))
		f.write('{test_file} macro_stance_f1:{macro_stance_f1}\n'.format(test_file=test_file, macro_stance_f1=macro_stance_f1))
	f.write('\n')
	f.close()