from __future__ import print_function
import argparse
import os
import shutil
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
from scipy.ndimage import uniform_filter1d
# import scipy as sp
from scipy import signal
import pickle
from utils.utils import Normalize
from utils.utils import calc_scores
import logging
# import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
from EvaluationMetrics.cccmetric import ccc

import math
from losses.CCC import CCC
#import wandb


def Test(val_loader, model, cam):
	# switch to evaluate mode
	global Val_acc
	global best_Val_acc
	global best_Val_acc_epoch
	#model.eval()
	model.eval()
	cam.eval()

	PrivateTest_loss = 0
	correct = 0
	total = 0
	running_val_loss = 0
	running_val_accuracy = 0

	vout = []
	vtar = []
	aout = []
	atar = []
	#torch.cuda.synchronize()
	#t7 = time.time()
	pred_a = dict()
	pred_v = dict()
	label_a = dict()
	label_v = dict()
	#files_dict = {}
	count = 0
	for batch_idx, (visualdata, audiodata, frame_ids, videos, vid_lengths) in tqdm(enumerate(val_loader),
														 total=len(val_loader), position=0, leave=True):
		#if(batch_idx > 2):#int(65844/64)):
		#	break

		#torch.cuda.synchronize()
		#t8 = time.time()
		#print('data loading time', t8-t7)

		audiodata = audiodata.cuda()#.unsqueeze(2)
		visualdata = visualdata.cuda()

		#torch.cuda.synchronize()
		#t9 = time.time()

		with torch.no_grad():
			b, seq_t, c, subseq_t, h, w = visualdata.size()

			#sub_seq_len = 16
			#visualdata = visual_data.view(b, c, -1, sub_seq_len, h, w)
			#visual_feats = []
			#aud_feats = []
			visual_feats = torch.empty((b, seq_t, 25088), dtype=visualdata.dtype, device = visualdata.device)
			aud_feats = torch.empty((b, seq_t, 512), dtype=visualdata.dtype, device = visualdata.device)
			for i in range(visualdata.shape[0]):
				#vis_dat = visualdata[i, :, :, :,:,:].transpose(0,1)
				audio_feat, visualfeat = model(audiodata[i,:,:,:], visualdata[i, :, :, :,:,:])
				visual_feats[i,:,:] = visualfeat
				aud_feats[i,:,:] = audio_feat

			audiovisual_vouts,audiovisual_aouts = cam(aud_feats, visual_feats)
			#outputs = audiovisual_outs.view(-1, audiovisual_outs.shape[0]*audiovisual_outs.shape[1])
			#targets = labels.view(-1, labels.shape[0]*labels.shape[1]).cuda()
			audiovisual_vouts = audiovisual_vouts.detach().cpu().numpy()
			audiovisual_aouts = audiovisual_aouts.detach().cpu().numpy()


			for voutputs, aoutputs, frameids, video, vid_length in zip(audiovisual_vouts, audiovisual_aouts, frame_ids, videos, vid_lengths):
				for voutput, aoutput, frameid, vid, length in zip(voutputs, aoutputs, frameids, video, vid_length):
					if vid not in pred_a:
						if frameid>1:
							print(vid)
							print(length)
							print("something is wrong")
							sys.exit()
						count = count + 1
						#files_dict[vid] = [0]*length

						pred_a[vid] = [0]*length
						pred_v[vid] = [0]*length

						#files_dict[vid][frameid-1] = [voutput, aoutput, labV, labA]
						pred_a[vid][frameid-1] = aoutput
						pred_v[vid][frameid-1] = voutput

					else:
						if frameid <= length:

							#print(frameid)
							#files_dict[vid][frameid-1] = [torch.tanh(output), lab]
							#pred_a[vid][frameid-1] = [voutput, aoutput, labV, labA]
							#files_dict[vid][frameid-1] = [voutput, aoutput, labV, labA]
							pred_a[vid][frameid-1] = aoutput
							pred_v[vid][frameid-1] = voutput

	if not os.path.isdir("results"):
		os.makedirs("results")
	for key in pred_a.keys():
		label_file_path = os.path.join('results', key + ".txt")
		text_file = open(label_file_path, "w")

		clipped_preds_v = np.clip(pred_v[key], -1.0, 1.0)
		clipped_preds_a = np.clip(pred_a[key], -1.0, 1.0)

		smoothened_preds_v = uniform_filter1d(clipped_preds_v, size=20, mode='constant')
		smoothened_preds_a = uniform_filter1d(clipped_preds_a, size=50, mode='constant')

		n = text_file.write("valence,arousal")
		n = text_file.write('\n')

		for i in range(len(smoothened_preds_a)):
			#vout.append(np.clip(smoothened_preds_v[i], -1.0, 1.0))
			#aout.append(np.clip(smoothened_preds_a[i], -1.0, 1.0))
			str_data = ','.join([str("{0:.5f}".format(smoothened_preds_v[i])), str("{0:.5f}".format(smoothened_preds_a[i]))])
			n = text_file.write(str_data)
			n = text_file.write('\n')
		text_file.close()
	sys.exit()
