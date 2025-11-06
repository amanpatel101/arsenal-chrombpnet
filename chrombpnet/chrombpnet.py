# Author: Lei Xiong <jsxlei@gmail.com>

import torch
import torch.nn as nn

from .bpnet import BPNet, DoubleBPNet, DreamRNN
import os
import sys
arsenal_dir = os.environ.get("ARSENAL_MODEL_DIR", "")
sys.path.append(f"{arsenal_dir}/src/regulatory_lm/")
from modeling.model import *


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

		self.tf_style_reinit()

	def tf_style_reinit(self):
		"""
		Re-initializes model weights for Linear and Conv1d layers using
		TensorFlow's default: Xavier/Glorot uniform for weights, zeros for bias.
		Operates in-place!
		"""
		print("Reinitializing with TF strategy")
		for m in self.model.modules():
			if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
				if hasattr(m, 'weight') and m.weight is not None:
					nn.init.xavier_uniform_(m.weight)
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.zeros_(m.bias)

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
		# counts_cat = torch.cat((acc_counts, bias_counts), dim=-1)
		# y_counts = torch.logsumexp(counts_cat, dim=-1)
		
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
		self.arsenal_model = arsenal_model
		if not self.finetune:
			for param in self.arsenal_model.parameters():
				param.requires_grad = False
			self.arsenal_model.eval()

		self.seq_input_size = config.input_len
		self.arsenal_input_size = config.arsenal_input_size
		self.softmax = torch.nn.Softmax(dim=-1)

		# Store category as buffer so it moves with .to()
		if config.category is not None:
			category_tensor = torch.tensor([config.category], dtype=torch.long)
			# register buffer named 'category'; access via self.category
			self.register_buffer("category", category_tensor)
		else:
			# no category provided
			self.category = None
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

		self.tf_style_reinit()

	def tf_style_reinit(self):
		"""
		Re-initializes model weights for Linear and Conv1d layers using
		TensorFlow's default: Xavier/Glorot uniform for weights, zeros for bias.
		Operates in-place!
		"""
		print("Reinitializing with TF strategy")
		for m in self.model.modules():
			if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
				if hasattr(m, 'weight') and m.weight is not None:
					nn.init.xavier_uniform_(m.weight)
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.zeros_(m.bias)


	def one_hot_to_tokens(self, X):
		tokens = torch.argmax(X, dim=-1)
		n_mask = (X.sum(dim=-1) == 0)
		tokens[n_mask] = 4
		return tokens

	def get_avg_embeddings(self, X, n=4):
		layers = [m for m in self.arsenal_model.modules() if type(m) in [TransformerROPEEncoderLayer, torch.nn.TransformerEncoderLayer]]
		# Pick last n layers (or adjust layer-type filter as needed)
		target_layers = layers[-n:]
		activations = []
		hooks = []
		
		def hook_fn(module, input, output):
			activations.append(output)
		
		# Register hooks
		hooks = [layer.register_forward_hook(hook_fn) for layer in target_layers]
		
		# Do inference (forward pass)
		_ = self.arsenal_model(X, self.category)
		
		# Remove hooks to avoid memory leaks
		for h in hooks:
			h.remove()
		
		return sum(activations) / len(activations)


	def normalize_probs(self, probs, tokens):
		#Calculate entropy-weighted probabilities
		# keep last dim so it broadcasts over the 4-class channel; clamp for numerical stability
		# entropy_metric = 2 + (probs * torch.log2(probs)).sum(-1, keepdim=True)
		# probs_norm = entropy_metric * probs
		probs_norm = probs
		onehot_start = torch.ones_like(probs)
		#Zero out everywhere except true indices
		idx = tokens.unsqueeze(-1)
		clamped = idx.clamp(0, 3)
		valid = (tokens >= 0) & (tokens < 4)
		kept = onehot_start.gather(2, clamped)
		kept = kept.masked_fill(~valid.unsqueeze(-1), 0)
		onehot_final = torch.zeros_like(onehot_start)
		onehot_final.scatter_(2, clamped, kept)
		probs_out = torch.cat((onehot_final, probs_norm), dim=-1)
		# probs_out = probs_norm
		return probs_out

	def combine_embeddings_onehot(self, embs, tokens):
		#Calculate entropy-weighted probabilities
		# keep last dim so it broadcasts over the 4-class channel; clamp for numerical stability
		# entropy_metric = 2 + (probs * torch.log2(probs)).sum(-1, keepdim=True)
		# probs_norm = entropy_metric * probs
		onehot_start = torch.ones(size=[embs.shape[0], embs.shape[1], 4], device=embs.device)
		#Zero out everywhere except true indices
		idx = tokens.unsqueeze(-1)
		clamped = idx.clamp(0, 3)
		valid = (tokens >= 0) & (tokens < 4)
		kept = onehot_start.gather(2, clamped)
		kept = kept.masked_fill(~valid.unsqueeze(-1), 0)
		onehot_final = torch.zeros_like(onehot_start)
		onehot_final.scatter_(2, clamped, kept)
		embs_out = torch.cat((onehot_final, embs), dim=-1)
		# probs_out = probs_norm
		return embs_out


	def get_likelihoods(self, tokens):
		if self.seq_input_size == self.arsenal_input_size:
			logits = self.arsenal_model(tokens, self.category)
			probs = self.softmax(logits)
			return self.normalize_probs(probs, tokens)
		#Else case - adapting to different input sizes
		#We basically break up the sequence into chunks of the model input length
		#Any remaining tokens are added by predicting the very end of the sequence and only concatenating the logits for previously unpredicted tokens
		else:
			full_partitions, remainder = self.seq_input_size // self.arsenal_input_size, self.seq_input_size % self.arsenal_input_size
			for part in range(full_partitions):
				curr_tokens = tokens[:,part * self.arsenal_input_size : part * self.arsenal_input_size + self.arsenal_input_size]
				if part == 0:
					logits = self.arsenal_model(curr_tokens, self.category)
					probs = self.normalize_probs(self.softmax(logits), curr_tokens)
				else:
					curr_logits = self.arsenal_model(curr_tokens, self.category)
					curr_probs = self.normalize_probs(self.softmax(curr_logits), curr_tokens)
					probs = torch.cat((probs, curr_probs), dim=1)
			#To account for the stragglers, we predict the very end of the sequence but only concatenate the stragglers
			final_pred = self.arsenal_model(tokens[:,-1*self.arsenal_input_size:], self.category)
			final_probs = self.normalize_probs(self.softmax(final_pred), tokens[:,-1*self.arsenal_input_size:])
			probs = torch.cat((probs, final_probs[:,-1*remainder:]), dim=1)

		return probs

	def get_embeddings(self, tokens):
		if self.seq_input_size == self.arsenal_input_size:
			embs = self.get_avg_embeddings(tokens)
		#Else case - adapting to different input sizes
		#We basically break up the sequence into chunks of the model input length
		#Any remaining tokens are added by predicting the very end of the sequence and only concatenating the embeddings for previously unpredicted tokens
		else:
			full_partitions, remainder = self.seq_input_size // self.arsenal_input_size, self.seq_input_size % self.arsenal_input_size
			for part in range(full_partitions):
				curr_tokens = tokens[:,part * self.arsenal_input_size : part * self.arsenal_input_size + self.arsenal_input_size]
				if part == 0:
					embs = self.get_avg_embeddings(curr_tokens)
				else:
					curr_embs = self.get_avg_embeddings(curr_tokens)
					embs = torch.cat((embs, curr_embs), dim=1)
			#To account for the stragglers, we predict the very end of the sequence but only concatenate the stragglers
			final_pred = self.get_avg_embeddings(tokens[:,-1*self.arsenal_input_size:])
			embs = torch.cat((embs, final_pred[:,-1*remainder:]), dim=1)

		return embs

	# def get_embeddings(self, tokens):
	# 	if self.seq_input_size == self.arsenal_input_size:
	# 		embs = self.arsenal_model.embed(tokens, self.category)
	# 	#Else case - adapting to different input sizes
	# 	#We basically break up the sequence into chunks of the model input length
	# 	#Any remaining tokens are added by predicting the very end of the sequence and only concatenating the embeddings for previously unpredicted tokens
	# 	else:
	# 		full_partitions, remainder = self.seq_input_size // self.arsenal_input_size, self.seq_input_size % self.arsenal_input_size
	# 		for part in range(full_partitions):
	# 			curr_tokens = tokens[:,part * self.arsenal_input_size : part * self.arsenal_input_size + self.arsenal_input_size]
	# 			if part == 0:
	# 				embs = self.arsenal_model.embed(curr_tokens, self.category)
	# 			else:
	# 				curr_embs = self.arsenal_model.embed(curr_tokens, self.category)
	# 				embs = torch.cat((embs, curr_embs), dim=1)
	# 		#To account for the stragglers, we predict the very end of the sequence but only concatenate the stragglers
	# 		final_pred = self.arsenal_model.embed(tokens[:,-1*self.arsenal_input_size:], self.category)
	# 		embs = torch.cat((embs, final_pred[:,-1*remainder:]), dim=1)

	# 	embs = self.batchnorm(embs.transpose(1,2)).transpose(1,2)
	# 	# embs = self.combine_embeddings_onehot(embs, tokens)
	# 	return embs


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
			x_embs = self.get_likelihoods(tokens)
			if x_embs.shape[-1] == 4:
				x_embs = x_embs.transpose(1,2)


		acc_profile, acc_counts = self.model(x_embs)
		bias_profile, bias_counts = self.bias(x)

		y_profile = acc_profile + bias_profile
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
		# counts_cat = torch.cat((acc_counts, bias_counts), dim=-1)
		# y_counts = torch.logsumexp(counts_cat, dim=-1)
		
		# DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
		return y_profile.squeeze(1), y_counts #.squeeze() 
	

class ArsenalChromBPNetNoBias(ArsenalChromBPNet):
	def __init__(self, 
		config,
		arsenal_model,
		**kwargs
		):
		super().__init__(config, arsenal_model, **kwargs)

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
			x_embs = self.get_likelihoods(tokens)
			if x_embs.shape[-1] == 4:
				x_embs = x_embs.transpose(1,2)



		acc_profile, acc_counts = self.model(x_embs)

		y_profile = acc_profile 
		y_counts = acc_counts
		
		# DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
		return y_profile.squeeze(1), y_counts #.squeeze() 
