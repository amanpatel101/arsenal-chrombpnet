# Author: Lei Xiong <jsxlei@gmail.com>

import torch
import torch.nn as nn

from .bpnet import BPNet


class _Exp(nn.Module):
	def __init__(self):
		super(_Exp, self).__init__()

	def forward(self, X):
		return torch.exp(X)


class _Log(nn.Module):
	def __init__(self):
		super(_Log, self).__init__()

	def forward(self, X):
		return torch.log(X)
	 

def _to_numpy(tensor):
	return tensor.detach().cpu().numpy()

# adapt from BPNet in bpnet-lite, credit goes to Jacob Schreiber <jmschreiber91@gmail.com>
class ChromBPNet(nn.Module):
	"""A ChromBPNet model.

	ChromBPNet is an extension of BPNet to handle chromatin accessibility data,
	in contrast to the protein binding data that BPNet handles. The distinction
	between these data types is that an enzyme used in DNase-seq and ATAC-seq
	experiments itself has a soft sequence preference, meaning that the
	strength of the signal is driven by real biology but that the exact read
	mapping locations are driven by the soft sequence bias of the enzyme.

	ChromBPNet handles this by treating the data using two models: a bias
	model that is initially trained on background (non-peak) regions where
	the bias dominates, and an accessibility model that is subsequently trained
	using a frozen version of the bias model. The bias model learns to remove
	the enzyme bias so that the accessibility model can learn real motifs.


	Parameters
	----------
	bias: torch.nn.Module 
		This model takes in sequence and outputs the shape one would expect in 
		ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
		from the bpnet-lite repo that has been trained on GC-matched non-peak
		regions.

	accessibility: torch.nn.Module
		This model takes in sequence and outputs the accessibility one would 
		expect due to the components of the sequence, but also takes in a cell 
		representation which modifies the parameters of the model, hence, 
		"dynamic." This model is usually a DynamicBPNet model, defined below.

	name: str
		The name to prepend when saving the file.
	"""

	def __init__(self, 
		config,
		**kwargs
		):
		super().__init__()

		self.model = BPNet(        
			out_dim=config.out_dim,
			n_filters=config.n_filters, 
			n_layers=config.n_layers, 
			conv1_kernel_size=config.conv1_kernel_size,
			profile_kernel_size=config.profile_kernel_size,
			n_outputs=config.n_outputs, 
			n_control_tracks=config.n_control_tracks, 
			profile_output_bias=config.profile_output_bias, 
			count_output_bias=config.count_output_bias, 
		)

		self.bias = BPNet(out_dim=config.out_dim, n_layers=4, n_filters=128)

		self._log = _Log()
		self._exp1 = _Exp()
		self._exp2 = _Exp()
		
		self.n_control_tracks = config.n_control_tracks

	def forward(self, x, **kwargs):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		x: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		X_ctl: ignore
			An ignored parameter for consistency with attribution functions.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, 1000)
			The predicted logit profile for each example. Note that this is not
			a normalized value.
		"""
		acc_profile, acc_counts = self.model(x)
		bias_profile, bias_counts = self.bias(x)

		y_profile = acc_profile + bias_profile
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
		
		# DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
		return y_profile.squeeze(1), y_counts #.squeeze() 
	

class ArsenalChromBPNet(nn.Module):
	"""A ChromBPNet model.

	ChromBPNet is an extension of BPNet to handle chromatin accessibility data,
	in contrast to the protein binding data that BPNet handles. The distinction
	between these data types is that an enzyme used in DNase-seq and ATAC-seq
	experiments itself has a soft sequence preference, meaning that the
	strength of the signal is driven by real biology but that the exact read
	mapping locations are driven by the soft sequence bias of the enzyme.

	ChromBPNet handles this by treating the data using two models: a bias
	model that is initially trained on background (non-peak) regions where
	the bias dominates, and an accessibility model that is subsequently trained
	using a frozen version of the bias model. The bias model learns to remove
	the enzyme bias so that the accessibility model can learn real motifs.


	Parameters
	----------
	bias: torch.nn.Module 
		This model takes in sequence and outputs the shape one would expect in 
		ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
		from the bpnet-lite repo that has been trained on GC-matched non-peak
		regions.

	accessibility: torch.nn.Module
		This model takes in sequence and outputs the accessibility one would 
		expect due to the components of the sequence, but also takes in a cell 
		representation which modifies the parameters of the model, hence, 
		"dynamic." This model is usually a DynamicBPNet model, defined below.

	name: str
		The name to prepend when saving the file.
	"""

	def __init__(self, 
		config,
		arsenal_model,
		**kwargs
		):
		super().__init__()

		self.finetune = config.finetune_arsenal
		self.arsenal_output_type = config.arsenal_output_type
		####ADD FINETUNE LOGIC HERE########
		if not self.finetune:
			self.arsenal_model = arsenal_model
			for param in self.arsenal_model.parameters():
				param.requires_grad = False
			self.arsenal_model.eval()

		self.seq_input_size = config.input_len
		self.arsenal_input_size = config.arsenal_input_size
		self.softmax = torch.nn.Softmax(dim=-1)

		#Need to adjust first convolutional layer
		self.model = BPNet(        
			out_dim=config.out_dim,
			n_filters=config.n_filters, 
			n_layers=config.n_layers, 
			conv1_kernel_size=config.conv1_kernel_size,
			profile_kernel_size=config.profile_kernel_size,
			n_outputs=config.n_outputs, 
			n_control_tracks=config.n_control_tracks, 
			profile_output_bias=config.profile_output_bias, 
			count_output_bias=config.count_output_bias, 
		)

		self.model.iconv = torch.nn.Conv1d(config.input_embedding_dim, config.n_filters, kernel_size=config.conv1_kernel_size, padding='valid')

		self.bias = BPNet(out_dim=config.out_dim, n_layers=4, n_filters=128)

		self._log = _Log()
		self._exp1 = _Exp()
		self._exp2 = _Exp()
		
		self.n_control_tracks = config.n_control_tracks

	def one_hot_to_tokens(self, X):
		tokens = torch.argmax(X, dim=-1)
		n_mask = (X.sum(dim=-1) == 0)
		tokens[n_mask] = 4
		return tokens

	def get_likelihoods(self, tokens):
		logits = self.arsenal_model(tokens, None)
		probs = self.softmax(logits)
		return probs

	def get_embeddings(self, tokens):
		if self.seq_input_size == self.arsenal_input_size:
			embs = self.arsenal_model.embed(tokens, None)
		#Else case - adapting to different input sizes
		#We basically break up the sequence into chunks of the model input length
		#Any remaining tokens are added by predicting the very end of the sequence and only concatenating the embeddings for previously unpredicted tokens
		else:
			full_partitions, remainder = self.seq_input_size // self.arsenal_input_size, self.seq_input_size % self.arsenal_input_size
			for part in range(full_partitions):
				curr_tokens = tokens[:,part * self.arsenal_input_size : part * self.arsenal_input_size + self.arsenal_input_size]
				if part == 0:
					embs = self.arsenal_model.embed(curr_tokens, None)
				else:
					curr_embs = self.arsenal_model.embed(curr_tokens, None)
					embs = torch.cat((embs, curr_embs), dim=1)
			#To account for the stragglers, we predict the very end of the sequence but only concatenate the stragglers
			final_pred = self.arsenal_model.embed(tokens[:,-1*self.arsenal_input_size:], None)
			embs = torch.cat((embs, final_pred[:,-1*remainder:]), dim=1)

		return embs


	def forward(self, x, **kwargs):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		x: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		X_ctl: ignore
			An ignored parameter for consistency with attribution functions.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, 1000)
			The predicted logit profile for each example. Note that this is not
			a normalized value.
		"""

		tokens = self.one_hot_to_tokens(x.transpose(1,2))
		if self.arsenal_output_type == "embedding":
			x_embs = self.get_embeddings(tokens) #We don't transpose because bpnet code does it for us if dimension is not 4
		elif self.arsenal_output_type == "likelihood":
			x_embs = self.get_likelihoods(tokens).transpose(1,2)


		acc_profile, acc_counts = self.model(x_embs)
		bias_profile, bias_counts = self.bias(x)

		y_profile = acc_profile + bias_profile
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
		
		# DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
		return y_profile.squeeze(1), y_counts #.squeeze() 