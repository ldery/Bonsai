import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
import math
from scipy.special import comb
import pdb
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
from torch.nn.init import kaiming_normal_
from scipy.stats import kendalltau
from collections import Counter
import matplotlib.pyplot as plt
import itertools
import gc

EPS = 1e-10


def split_train_test(run_info, normalization, num_folds=5, sid=0, no_split=False):
	# Split into train and test
	train_frac = 1.0 - (1.0 / num_folds)
	xs, ys = run_info

	if no_split:
		trn_list, tst_list = list(range(len(ys))), list(range(len(ys)))
	else:
		np.random.seed(0)
		perm = np.random.permutation(len(ys))
		bsz = len(ys) // num_folds
		tst_list = set(range(sid * bsz, (sid + 1)* bsz, 1))
		trn_list = set(range(len(ys))) - tst_list

	tr_xs, tr_ys = [xs[i] / normalization for i in trn_list], [ys[i] for i in trn_list]
	tr_ys  = torch.tensor(tr_ys).float().cuda()
	mean_ys = tr_ys.mean().item()
	tr_ys -= mean_ys

	ts_xs, ts_ys = [xs[i] / normalization for i in tst_list], [ys[i] for i in tst_list]
	ts_ys  = torch.tensor(ts_ys).float().cuda()
	ts_ys -= mean_ys

	return  (tr_xs, tr_ys), (ts_xs, ts_ys)

# Module for doing hyper-param search over the score models
class ScoreModelHP(object):
	def __init__(
		self, id_='sm_model', num_players=None, base_mask=None, 
		hp_dict=None, wandb=None
	):
		self.id_ = id_
		self.num_players = num_players
		self.base_mask = base_mask
		self.hp_dict = hp_dict
		self.wandb = wandb

		self.best_fit_model = None
		self.best_fit_stats = None
		self.best_hp = None

	def search_best_linear_fit(self, run_info):
		normalization = self.base_mask.sum().item() + 1.0 # in case there is division by zero
		normalization = np.sqrt(normalization)

		keys = self.hp_dict.keys()
		values = self.hp_dict.values()
		hp_list = list(itertools.product(*values))

		num_folds = 5
		best_fit_tau = float("-inf")
		for hp_set in hp_list:
			this_hp = {k:v for k, v in zip(keys, hp_set)}

			all_ktaus = []
			for split_id in range(num_folds):
				this_run_info = split_train_test(run_info, normalization, num_folds=num_folds, sid=split_id) 
				sm = ScoreModel(id_=self.id_, num_players=self.num_players, base_mask=self.base_mask, hp_dict=this_hp)
				sm.cuda()
				results_info = sm.update_with_info(this_run_info, normalization)
				all_ktaus.append(results_info[0])
				del sm

			avg_ktau = np.mean(all_ktaus)
			if avg_ktau > best_fit_tau:
				best_fit_tau = avg_ktau
				self.best_hp = this_hp


		sm = ScoreModel(id_=self.id_, num_players=self.num_players, base_mask=self.base_mask, hp_dict=self.best_hp)
		sm.cuda()
		this_run_info = split_train_test(run_info, normalization, no_split=True) 
		results_info = sm.update_with_info(this_run_info, normalization)
		self.best_fit_model = (sm.base_model.score_tensor.clone().squeeze()).detach()
		self.best_fit_stats = results_info

		del sm

		# do some wandb logging here with self.best_fit_stats
		if self.wandb is not None:
			self.plot_best()
			self.wandb.log({"best-details/{}".format(self.id_): self.best_hp})

		gc.collect()
		torch.cuda.empty_cache()

	def get_best_fit(self):
		return self.best_fit_model

	def plot_best(self):
		wandb_dict = self.best_fit_stats[-1]
		fig, ax = plt.subplots(1, 2, figsize=(10, 20))
		for k, v in wandb_dict.items():
			if 'loss_avg' in k:
				ax[0].plot(v, label=k)
			elif 'kendall' in k:
				ax[1].plot(v, label=k)
		ax[0].legend()
		ax[1].legend()
		fig.tight_layout()
		fig.suptitle('Our Tau = {} | Naive Tau = {}'.format(self.best_fit_stats[0], self.best_fit_stats[1]))

		self.wandb.log({"{}_chart".format(self.id_): fig})

		data = self.best_fit_model.cpu().numpy()
		try:
			fig, ax = plt.subplots(figsize=(20, 20))
			frq, edges = np.histogram(data, bins=20)
			ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
			ax.set_xlabel('Index Weight')
			ax.set_ylabel('Frequency')
			self.wandb.log({"{}_weight-histogram".format(self.id_): fig})
		except:
			self.wandb.log({"{}_weight-info".format(self.id_): {'min': np.min(data), 'max': np.max(data), 'mean': np.mean(data), 'std': np.std(data)}})


def create_tensor(shape, zero_out=1.0, requires_grad=True, is_cuda=True):
	inits = torch.zeros(*shape)
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights

def get_score_model(config, num_layers, num_players, model_type, reg_weight, wandb):
	return ScoreModel(config, num_layers=num_layers, num_players=num_players, reg_weight=reg_weight, wandb=wandb)


class LinearModel(nn.Module):
	def __init__(self, num_players, base_mask, reg_weight=None, reg_type='l1'):
		super(LinearModel, self).__init__()
		self.score_tensor = nn.parameter.Parameter(create_tensor((num_players, 1)))
		self.score_bias = nn.parameter.Parameter(create_tensor((1,)))
		self.num_players = num_players
		self.base_mask = base_mask
		self.reg_weight = reg_weight
		self.reg_type = reg_type

		# Initialize to the base mask
		with torch.no_grad():
			self.score_tensor.add_(base_mask)

	def set_reg_weight(self, reg_weight):
		self.reg_weight = reg_weight

	def regLoss(self, mean_score_tensor):
		if self.reg_type == 'l1':
			l1reg = self.score_tensor.abs().sum() * self.reg_weight
			return l1reg

		mean_score_tensor = 0.0 if mean_score_tensor is None else mean_score_tensor
		weighted_l2 = ((self.score_tensor - mean_score_tensor)**2)  * self.reg_weight
		return weighted_l2.sum()

	def forward(self, xs):
		return torch.matmul(xs, self.score_tensor) + self.score_bias

	def get_scores(self, base_mask):
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			active_preds = predictions[base_mask > 0]

		stats = active_preds.mean().item(), active_preds.std().item(), active_preds.max().item(), active_preds.min().item()
		print("Predictions vals : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

		return predictions


class ScoreModel(nn.Module):
	def __init__(self, id_='sm_model', num_players=None, base_mask=None, hp_dict=None, wandb=None):
		super(ScoreModel, self).__init__()

		assert hp_dict is not None, 'List of hyper-parameters have not been specified'
		self.hp_dict = hp_dict
		self.base_model = LinearModel(num_players, base_mask, reg_weight=self.hp_dict['reg_weight'], reg_type=self.hp_dict['reg_type'])
		self.loss_fn = nn.MSELoss()
		self.candidate_buffer = []
		self.curr_norm = 1.0
		self.wandb = wandb
		self.id_ = id_

	def set_prior_variance(self, prior_variance):
		self.prior_variance = prior_variance
		self.base_model.set_reg_weight(prior_variance)

	def forward(self, xs):
		return self.base_model.forward(xs)

	def run_epoch(self, epoch_, xs, ys, mean=None, is_train=True):
		# generate a permutation
		perm = np.random.permutation(len(ys))
		running_loss_avg, running_reg_avg = 0.0, 0.0
		n_batches = math.ceil(len(ys) / self.hp_dict['bsz'])
		if not is_train:
			self.base_model.eval()
		max_error, min_error = -1, 1
		all_preds, all_ys = [], []
		for batch_id in range(n_batches):
			start_, end_ = int(self.hp_dict['bsz'] * batch_id), int(self.hp_dict['bsz'] * (batch_id + 1))
			xs_masks = torch.stack([xs[i_] for i_ in perm[start_:end_]]).float().cuda()

			this_ys = ys[perm[start_:end_]].view(-1, 1)
			# do the forward pass heres
			preds = self.forward(xs_masks)

			loss = self.loss_fn(preds, this_ys)
			regLoss = self.base_model.regLoss(mean)

			all_ys.append(this_ys.cpu().numpy())
			all_preds.append(preds.detach().cpu().numpy())

			# Do some logging here
			with torch.no_grad():
				errors = (preds - this_ys).abs()
				this_max_error = errors.max().item()
				this_min_error = errors.min().item()

			max_error = min(max_error, this_max_error)
			min_error = min(min_error, this_min_error)

			running_loss_avg += loss.item()
			running_reg_avg += regLoss.item()
			if is_train:
				# Clamp the losses to be within bounds of the training data likelihood.
				loss = loss + regLoss
				loss.backward()

				self.score_optim.step()
				self.score_optim.zero_grad()

			
		running_loss_avg /= n_batches
		running_reg_avg /= n_batches
		if not is_train:
			self.base_model.train()
		# Compute the kendalltau statistic
		kendall_tau = kendalltau(np.concatenate(all_preds), np.concatenate(all_ys)).correlation
		return running_loss_avg, max_error, min_error, kendall_tau


	def update_with_info(self, run_info, normalization):

		self.set_prior_variance(self.hp_dict['reg_weight'] / normalization)
		self.curr_norm = normalization

		(tr_xs, tr_ys), (ts_xs, ts_ys) = run_info

		with torch.no_grad():
			mean_score_tensor = self.base_model.score_tensor.clone().detach()
			if mean_score_tensor.sum() == 0:
				mean_score_tensor = torch.stack(tr_xs).mean(axis=0).view(-1, 1)
			mean_score_tensor.requires_grad = False

		# Setup the optimizer and learn the new model
		lr =  1.0/(len(mean_score_tensor.nonzero()) * self.hp_dict['lr_factor'])

		self.score_optim = Adam(self.base_model.parameters(), lr=lr)
		lr_scheduler = ReduceLROnPlateau(self.score_optim, mode='max', factor=0.5, patience=3, min_lr=1e-5)

		best_tau, clone, since_best, best_run = -1e5, None, 0, None
		wandb_dict = {
			'tr/loss_avg': [], 'ts/loss_avg': [],
			'tr/max_err': [], 'ts/max_err': [],
			'tr/min_err': [], 'ts/min_err': [],
			'tr/kendall': [], 'ts/kendall': []
		}
		for epoch_ in range(self.hp_dict['nepochs']):

			tr_run_out = self.run_epoch(epoch_, tr_xs, tr_ys, mean=mean_score_tensor)
			wandb_dict['tr/loss_avg'].append(tr_run_out[0])
			wandb_dict['tr/max_err'].append(tr_run_out[1])
			wandb_dict['tr/min_err'].append(tr_run_out[2])
			wandb_dict['tr/kendall'].append(tr_run_out[3])

			ts_run_out = self.run_epoch(epoch_, ts_xs, ts_ys, is_train=False)
			wandb_dict['ts/loss_avg'].append(ts_run_out[0])
			wandb_dict['ts/max_err'].append(ts_run_out[1])
			wandb_dict['ts/min_err'].append(ts_run_out[2])
			wandb_dict['ts/kendall'].append(ts_run_out[3])

			if ts_run_out[-1] >= best_tau:
				best_tau = ts_run_out[-1]
				best_run = ts_run_out
				clone = deepcopy(self.base_model)
				since_best = 0

			lr_scheduler.step(ts_run_out[-1]) # step based on the max error
			since_best += 1
			if since_best > self.hp_dict['patience']:
				break

		if clone is not None:
			del self.base_model
			self.base_model = clone

		with torch.no_grad():
			tr_xs = torch.stack(tr_xs).cuda()
			ts_xs = torch.stack(ts_xs).float().cuda()
			ts_ys = ts_ys.cpu().numpy()
			with torch.no_grad():
				preds = self.base_model.forward(ts_xs).squeeze()
				preds = preds.detach().cpu().numpy()

			base_pred = np.ones_like(ts_ys) * tr_ys.mean().item()
			base_coef_det = kendalltau(ts_ys, base_pred)

		return best_tau, base_coef_det.correlation, wandb_dict



