"""
PyTorch Child-Sum Tree LSTM model

See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.
"""

import torch


class TreeLSTM(torch.nn.Module):
	'''PyTorch TreeLSTM model that implements efficient batching.
	'''
	def __init__(self, in_features, out_features, hidden_units,hidden_stance_units, out_stance_features,dropout=0.1):
		'''TreeLSTM class initializer

		Takes in int sizes of in_features and out_features and sets up model Linear network layers.
		'''
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.hidden_units = hidden_units


		# bias terms are only on the W layers for efficiency
		self.W_iou = torch.nn.Linear(self.in_features, 3 * (self.hidden_units//16) * self.out_features)

		self.U_iou = torch.nn.Linear((self.hidden_units//16) * self.out_features, 3 * (self.hidden_units//16) * self.out_features, bias=False)

		# f terms are maintained seperate from the iou terms because they involve sums over child nodes
		# while the iou terms do not
		self.W_f = torch.nn.Linear(self.in_features, (self.hidden_units//16) * self.out_features)
		self.U_f = torch.nn.Linear((self.hidden_units//16) * self.out_features, (self.hidden_units//16) * self.out_features, bias=False)

		self.feedforward = torch.nn.Linear((self.hidden_units//16) * self.out_features, self.out_features, bias=False)
		self.stance_dense = torch.nn.Linear((self.hidden_units//16) * self.out_features, hidden_stance_units, bias=False)
		self.stance_dense2 = torch.nn.Linear(hidden_stance_units, (self.hidden_units//16) * self.out_features, bias=False)
		# self.stance_dense3 = torch.nn.Linear(1000, hidden_stance_units, bias=False)
		self.layer_norm = torch.nn.LayerNorm(out_stance_features, eps=1e-6)
		self.dropout = torch.nn.Dropout(dropout)
		self.stance_feedforward = torch.nn.Linear((self.hidden_units//16) * self.out_features, out_stance_features, bias=False)

		self.init_weights()

	def init_weights(self):
		for p in self.parameters():
			if p.data.ndimension() >= 2:
				torch.nn.init.xavier_uniform_(p.data)
			else:
				torch.nn.init.zeros_(p.data)


	def forward(self, features, node_order, adjacency_list, edge_order, root_node, root_label):
		'''Run TreeLSTM model on a tree data structure with node features

		Takes Tensors encoding node features, a tree node adjacency_list, and the order in which 
		the tree processing should proceed in node_order and edge_order.
		'''

		# Total number of nodes in every tree in the batch
		batch_size = node_order.shape[0]

		# Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
		device = next(self.parameters()).device

		# h and c states for every node in the batch
		h = torch.zeros(batch_size, (self.hidden_units//16) * self.out_features, device=device)
		
		c = torch.zeros(batch_size, (self.hidden_units//16) * self.out_features, device=device)

		# print(node_order.shape)

		# populate the h and c states respecting computation order
		for n in range(node_order.max() + 1):
			# print(n)
			self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)
		residual = h
		h_root = self.feedforward(h[root_node, :])
		h_stance = self.stance_dense(h)
		h_stance = self.dropout(h_stance)
		h_stance = self.stance_dense2(h_stance)
		# h_stance = self.stance_dense3(h_stance)
		h_stance = self.dropout(h_stance)
		h_stance = h_stance + residual
		h_stance = self.stance_feedforward(h_stance)
		h_stance = self.dropout(h_stance)
		h_stance = self.layer_norm(h_stance)

		h_root = torch.nn.functional.softmax(h_root, dim = 1)
		h_stance = torch.nn.functional.softmax(h_stance, dim=1)

		return h_stance, h_root, c

	def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
		'''Helper function to evaluate all tree nodes currently able to be evaluated.
		'''
		# N is the number of nodes in the tree
		# n is the number of nodes to be evaluated on in the current iteration
		# E is the number of edges in the tree
		# e is the number of edges to be evaluated on in the current iteration
		# F is the number of features in each node
		# M is the number of hidden neurons in the network

		# node_order is a tensor of size N x 1
		# edge_order is a tensor of size E x 1
		# features is a tensor of size N x F
		# adjacency_list is a tensor of size E x 2

		# node_mask is a tensor of size N x 1
		node_mask = node_order == iteration

		# print(node_mask)

		# edge_mask is a tensor of size E x 1
		edge_mask = edge_order == iteration

		# x is a tensor of size n x F
		x = features[node_mask, :]

		# x = self.hidden_layer(x_orig)

		# At iteration 0 none of the nodes should have children
		# Otherwise, select the child nodes needed for current iteration
		# and sum over their hidden states
		if iteration == 0:
			iou = self.W_iou(x)
		else:
			# adjacency_list is a tensor of size e x 2
			adjacency_list = adjacency_list[edge_mask, :]

			# parent_indexes and child_indexes are tensors of size e x 1
			# parent_indexes and child_indexes contain the integer indexes needed to index into
			# the feature and hidden state arrays to retrieve the data for those parent/child nodes.
			parent_indexes = adjacency_list[:, 0]
			child_indexes = adjacency_list[:, 1]

			# child_h and child_c are tensors of size e x 1
			child_h = h[child_indexes, :]
			child_c = c[child_indexes, :]

			# Add child hidden states to parent offset locations
			_, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
			child_counts = tuple(child_counts)

			parent_children = torch.split(child_h, child_counts)
			parent_list = [item.sum(0) for item in parent_children]

			h_sum = torch.stack(parent_list)
			iou = self.W_iou(x) + self.U_iou(h_sum)

		# i, o and u are tensors of size n x M
		i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
		i = torch.sigmoid(i)
		o = torch.sigmoid(o)
		u = torch.tanh(u)

		# At iteration 0 none of the nodes should have children
		# Otherwise, calculate the forget states for each parent node and child node
		# and sum over the child memory cell states
		if iteration == 0:
			c[node_mask, :] = i * u
		else:
			# f is a tensor of size e x M
			f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
			f = torch.sigmoid(f)

			# fc is a tensor of size e x M
			fc = f * child_c

			# Add the calculated f values to the parent's memory cell state
			parent_children = torch.split(fc, child_counts)
			parent_list = [item.sum(0) for item in parent_children]

			c_sum = torch.stack(parent_list)
			c[node_mask, :] = i * u + c_sum

		h[node_mask, :] = o * torch.tanh(c[node_mask])
