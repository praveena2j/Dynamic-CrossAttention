import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import librosa
import torchaudio
from torchvision import transforms
import torch
from scipy import signal
from .spec_transform import *
from .clip_transforms import *
import bisect
import cv2
import pandas as pd
import utils.videotransforms as videotransforms
import re
#from models.vggish_pytorch import vggish_input
import csv
import math

def get_filename(n):
	filename, ext = os.path.splitext(os.path.basename(n))
	return filename

def default_seq_reader(videoslist, win_length, stride, dilation, wavs_list):
	shift_length = stride #length-1
	sequences = []
	#csv_data_list = os.listdir(videoslist)#[39:]#[0:3]
	#csv_data_list = ['video94.csv'] #os.listdir(videoslist)#[39:]#[0:3]
	#csv_data_list = ['12-24-1920x1080.csv']
	csv_data_list = ['video92.csv']
	#csv_data_list = ['317.csv']
	#csv_data_list = ['21-24-1920x1080.csv'] #['video92.csv']
	print(csv_data_list)

	print("Number of Sequences: " + str(len(set(csv_data_list))))
	for video in csv_data_list:
		#video = '220.csv'
		if video.startswith('.'):
			continue
		vid_data = pd.read_csv(os.path.join(videoslist,video))
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		labels_A = video_data['A']
		label_arrayV = np.asarray(labels_V, dtype = np.float32)
		label_arrayA = np.asarray(labels_A, dtype = np.float32)
		#medfiltered_labels = signal.medfilt(label_array)
		frame_ids = video_data['frame_id']
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
		#timestamp_file = os.path.join(time_list, get_filename(video) +'_video_ts.txt')
		f_name = get_filename(video)
		if f_name.endswith('_left'):
			wav_file_path = os.path.join(wavs_list, f_name[:-5])
			vidname = f_name[:-5]
		elif f_name.endswith('_right'):
			wav_file_path = os.path.join(wavs_list, f_name[:-6])
			vidname = f_name[:-6]
		else:
			wav_file_path = os.path.join(wavs_list, f_name)
			vidname = f_name
		#label_array = np.asarray(labels_V, dtype=np.float32)
		#medfiltered_labels = signal.medfilt(label_array)
		vid = np.asarray(list(zip(images, label_arrayV, label_arrayA, frameid_array)))
		#f = open(timestamp_file)
		#time_lines = f.readlines()

		time_filename = os.path.join('datasets/realtimestamps', vidname) + '_video_ts.txt'
		f = open(os.path.join(time_filename))
		lines = f.readlines()[1:]
		length = len(lines) #len(os.listdir(wav_file_path))
		end = 481
		start = end -win_length
		#start = 0 #end - win_length
		#end = start + win_length
		counter = 0
		cnt = 0
		result = []
		#if end < length:
		while end < length + 482:
			avail_seq_length = end -start
			#sequence_length = win_length / dilation
			# Extracting the indices between the start and start + 128 (sequence length)
			#indices = np.arange(start, end, dilation) + (dilation -1)
			#indices = np.arange(math.ceil(sequence_length))
			#indices = np.flip(end - dilation*(np.arange(math.ceil(sequence_length))))
			#frame_id = frameid_array[indices]
			#print(frame_id)
			#indices = np.where((frameid_array>=start+1) & (frameid_array<=end))[0]
			count = 15
			#subseq_indices_check = []
			num_samples = 0
			vis_subsequnces = []
			aud_subsequnces = []
			for i in range(16):
				#subseq_indices.append(np.where((frameid_array>=((i-1)*32)+1) & (frameid_array<=32*i))[0])
				sub_indices = np.where((frameid_array>=(start+(i*32))+1) & (frameid_array<=(end -(count*32))))[0]
				wav_file = os.path.join(wav_file_path, str(end -(count*32))) +'.wav'

				#print(sub_indices)
				if (end -(count*32)) <= length:
					result.append(end -(count*32))
				if ((start+(i*32))+1) <0 and (end -(count*32)) <0:
					vis_subsequnces.append([])
				if len(sub_indices)>=8 and len(sub_indices)<16:
					subseq_indices = sub_indices[-8:]
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices)>=16 and len(sub_indices)<24:
					subseq_indices = np.flip(np.flip(sub_indices)[::2])
					subseq_indices = subseq_indices[-8:]
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices)>=24 and len(sub_indices)<32:
					subseq_indices = np.flip(np.flip(sub_indices)[::3])
					subseq_indices = subseq_indices[-8:]
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices) == 32:
					subseq_indices = np.flip(np.flip(sub_indices)[::4])
					vis_subsequnces.append([vid[subseq_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				elif len(sub_indices) > 0 and len(sub_indices) < 8:
					newList = [sub_indices[-1]]* (8-len(sub_indices))
					sub_indices = np.append(sub_indices, np.array(newList), 0)
					vis_subsequnces.append([vid[sub_indices], (end -(count*32)), f_name, length])
					aud_subsequnces.append(wav_file)
				else:
					vis_subsequnces.append([[], (end -(count*32)),f_name, length])
					aud_subsequnces.append(wav_file)

				count = count - 1

			if (len(aud_subsequnces) < 16):
				print(end -(count*32))
				print(aud_subsequnces)
				sys.exit()
			start_frame_id = start +1
			#wav_file = os.path.join(wav_file_path, str(end)) +'.wav'

			#if len(vis_subsequnces) == 16:
			#sequences.append([subsequnces, wav_file, start_frame_id])
			sequences.append([vis_subsequnces, aud_subsequnces])
			#else:
			#	sequences.append([])
			if avail_seq_length>512:
				print("Wrong Sequence")
				sys.exit()
			counter = counter + 1
			if counter > 31:
				end = end + 480 + shift_length
				start = end - win_length
				#start = start + 224 + shift_length
				#end = start + win_length
				counter = 0
			else:
				end = end + shift_length
				start = end - win_length

		result.sort()
		#print(result)
		#print("-----------")
		#print(len(set(result)))
		#print(length)
		#print(len(sequences))
		#return sequences
		if len(set(result)) == length:
			continue
		else:
			print(video)
			print(len(set(result)))
			print(length)
			print("Seq lengths are wrong")
			sys.exit()
		#	#sequences.append([seq, wav_file, start_frame_id])
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		#print(fileList)
		video_length = 0
		videos = []
		lines = list(file)
		#print(len(lines))
		for i in range(9):
			line = lines[video_length]
			#print(line)
			#line = file.readlines()[video_length + i]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			#print(find_str)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			#print(new_video_length)Visualmodel_for_Afwild2_bestworkingcode_avail_lab_img_videolevel_perf
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
			#print(video_length)
	return videos

def plot_spectrogram(spec, count, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
	fig, axs = plt.subplots(1, 1)
	#axs.set_title(title or 'Spectrogram (db)')
	#axs.set_ylabel(ylabel)
	#axs.set_xlabel('frame')
	im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
	if xmax:
		axs.set_xlim((0, xmax))
	#fig.colorbar(im, ax=axs)
	plt.axis('off')
	plt.savefig('spec'+ str(count) + '.png')
	plt.show(block=False)




class ImageList_val_vis(data.Dataset):
	def __init__(self, root, fileList, audList, length, flag, stride, dilation, subseq_length, list_reader=default_list_reader, seq_reader=default_seq_reader):
		self.root = root
		#self.label_path = label_path
		self.videoslist = fileList #list_reader(fileList)
		self.win_length = length
		#self.time_list = timestmps
		self.num_subseqs = int(self.win_length / subseq_length)
		self.wavs_list = audList
		self.stride = stride
		self.dilation = dilation
		self.subseq_length = int(subseq_length / self.dilation)
		self.sequence_list = seq_reader(self.videoslist, self.win_length, self.stride, self.dilation, self.wavs_list)
		#self.stride = stride
		self.sample_rate = 44100
		self.window_size = 20e-3
		self.window_stride = 10e-3
		self.sample_len_secs = 1
		self.sample_len_clipframes = int(self.sample_len_secs * self.sample_rate * self.num_subseqs)
		self.sample_len_frames = int(self.sample_len_secs * self.sample_rate)
		self.audio_shift_sec = 1
		self.audio_shift_samples = int(self.audio_shift_sec * self.sample_rate)

		#self.transform = transform
		#self.dataset = dataset
		#self.loader = loader
		self.flag = flag

	def __getitem__(self, index):
		#for video in self.videoslist:
		seq_path, wav_file = self.sequence_list[index]
		#seq_path = self.sequence_list[index]
		#img = self.loader(os.path.join(self.root, imgPath), self.flag)
		#if (self.flag == 'train'):
		seq, orig_seqs, fr_ids, video, vid_lengths, labelV, labelA = self.load_vis_data(self.root, seq_path, self.flag, self.subseq_length)
		aud_data,aud_data_orig = self.load_aud_data(wav_file, self.num_subseqs, self.flag)
		#label_index = torch.DoubleTensor([label])
		#else:
		#   seq, label = self.load_test_data_label(seq_path)
		#   label_index = torch.LongTensor([label])
		#if self.transform is not None:
		#    img = self.transform(img)
		return seq, orig_seqs, aud_data, fr_ids, video, vid_lengths, labelV, labelA, aud_data_orig#_index

	def __len__(self):
		return len(self.sequence_list)

	def load_vis_data(self, root, SeqPath, flag, subseq_len):
		#print("Loadung training data")
		clip_transform = ComposeWithInvert([NumpyToTensor(),
												 Normalize(mean=[0.43216, 0.394666, 0.37645],
														   std=[0.22803, 0.22145, 0.216989])])
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
					#transforms.RandomResizedCrop(224),
					#transforms.RandomHorizontalFlip(),
					#transforms.ToTensor(),
			])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224),
				#transforms.Resize(256),
				#transforms.CenterCrop(224),
				#transforms.ToTensor(),
			])
		output = []
		subseq_inputs = []
		subseq_labels = []
		labV = []
		labA = []
		frame_ids = []
		v_names = []
		v_lengths = []
		seq_length = math.ceil(self.win_length / self.dilation)
		seqs = []
		orig_seqs = []
		for clip in SeqPath:
			seq_clip = clip[0]
			frame_id = clip[1]
			v_name = clip[2]
			v_length = clip[3]
			images = np.zeros((8, 112, 112, 3), dtype=np.uint8)
			labelV = -5.0
			labelA = -5.0
			#inputs = []
			label_V = []
			label_A = []
			for im_index, image in enumerate(seq_clip):
				#if len(image)>1:
				imgPath = image[0]
				label_V.append(image[1])
				label_A.append(image[2])
				#labelV = image[1]
				#labelA = image[2]

				try:
					img = np.array(Image.open(os.path.join(root , imgPath)))
					images[im_index, :, :, 0:3] = img
				except:
					pass

				#	img = np.zeros((112, 112, 3), dtype=np.float32)
				#w,h,c = img.shape
				##w,h = img.size
				#if w == 0:
				#	continue
				#else:
				#	img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]
				#	#img = img.resize((256, 256), Image.ANTIALIAS)
				#img = (img/255.)*2 - 1
				#img = img.resize((256,256), Image.ANTIALIAS)
				#inputs.append(data_transforms(img).unsqueeze(0))
				#inputs.append(img)
			#imgs = data_transforms(images)
			#imgs=np.asarray(inputs, dtype=np.float32)

			labelV = label_V[0] #np.mean(np.array(label_V).astype(np.float))
			labelA = label_A[0] #np.mean(np.array(label_A).astype(np.float))
			imgs = clip_transform(images)
			seqs.append(imgs)
			orig_seqs.append(torch.from_numpy(images))
			#seqs.append(imgs)
			v_names.append(v_name)
			frame_ids.append(frame_id)
			v_lengths.append(v_length)
			#subseq_inputs.append(inputs)
			labV.append(float(labelV))
			labA.append(float(labelA))

			#lab.append(float(label))
			#if len(inputs) == subseq_len:
			#	subseq_inputs.append(inputs)
			#	lab.append(float(label))
			#	inputs = []

		targetsV = torch.FloatTensor(labV)
		targetsA = torch.FloatTensor(labA)
		imgframe_ids = np.stack(np.asarray(frame_ids))#.permute(4,0,1,2,3)

		#targets = torch.mean(label)
		#for subseq in subseq_inputs:
		#	imgs=np.asarray(subseq, dtype=np.float32)
		#	#if(imgs.shape[0] != 0):
		#	imgs = data_transforms(imgs)
		#	#seqs.append(imgs)
		#	seqs.append(torch.from_numpy(imgs))
		vid_seqs = torch.stack(seqs)#.permute(4,0,1,2,3)
		orig_vid_seqs = torch.stack(orig_seqs)#.permute(4,0,1,2,3)
		#vid_seqs = np.stack(seqs)#.permute(4,0,1,2,3)
		#for img, fm_id in zip(vid_seqs, imgframe_ids):
		#	#print(img.cpu().numpy()[:,7,:,:].shape)
		#	plt.imshow(img.cpu().numpy()[:,7,:,:])
		#	plt.savefig("test.png", bbox_inches = "tight", pad_inches = 0.0)
		#	sys.exit()

		#	print(fm_id)
		#	sys.exit()

		#print(imgframe_ids.shape)
		#print(vid_seqs.shape)
		#sys.exit()
		return vid_seqs, orig_vid_seqs, imgframe_ids, v_names, v_lengths, targetsV, targetsA # vid_seqs,

		#frame_ids.append(ids)

		#print(label)
		#label_idx = float(label)

		if (len(inputs) < int(seq_length)):
			imgs = np.zeros((seq_length, 224, 224, 3), dtype=np.int16)
			lables = np.zeros((self.win_length), dtype=np.int16)
			indices = np.arange(len(inputs))
			#imgs[indices[0]:indices[len(indices)-1]] = inputs
			imgs[indices] = inputs
			#lables[indices[0]:indices[len(indices)-1]+1] = lab
			imgs=np.asarray(imgs, dtype=np.float32)
			#targets = np.asarray(lables, dtype=np.float32)
		else:
			imgs=np.asarray(inputs, dtype=np.float32)
			#targets = np.asarray(lab, dtype=np.float32)

		#imgs=np.asarray(inputs, dtype=np.float32)
		#targets = np.asarray(lab, dtype=np.float32)
		#output_subset = torch.cat(inputs)#.unsqueeze(0)
		#output.append(output_subset)

		#if(imgs.shape[0] != 0):
		imgs = data_transforms(imgs)
		for i in range(0,imgs.shape[0], subseq_len):
			subseq_inputs.append(torch.from_numpy(imgs[i:i+subseq_len,:,:,:]))
		vid_seqs = torch.stack(subseq_inputs).permute(4,0,1,2,3)
		#	return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), lab
		#return output_subset, lab
		return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), float(targets)
		#return torch.from_numpy(imgs.transpose([0, 3, 1, 2])), targets
		#return torch.from_numpy(imgs), lab
		#return output_subset, lab
		#else:
		#	return [], []

	def load_aud_data(self, wav_file, num_subseqs, flag):
		transform_spectra = transforms.Compose([
			transforms.ToPILImage(),
			#transforms.Resize((224,224)),
			transforms.RandomVerticalFlip(1),
			transforms.ToTensor(),
		])
		audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

		#waveform, sr = torchaudio.load(wav_file)
		#subseq_len = waveform.shape[1] / num_subseqs
		spectrograms = []
		waves = []
		max_spec_shape = []
		if(len(wav_file) < 16):
			print(wav_file)
			sys.exit()

		for wave in wav_file:
			id = os.path.splitext(os.path.basename(wave))[0]
			#for i in range(int(num_subseqs)):
			if wave == []:
				audio = torch.zeros((1, 45599))
			elif not os.path.isfile(wave):
				audio = torch.zeros((1, 45599))
			else:
				try:
					audio, sr = torchaudio.load(wave) #,
									#num_frames=int(subseq_len),
									#frame_offset=int(subseq_len*i))
				except:
					audio, sr = torchaudio.load(wave) #,
			#x = vggish_input.waveform_to_examples(audio.numpy(), self.sample_rate)
			if audio.shape[1] <= 45599:
				_audio = torch.zeros((1, 45599))
				_audio[:, -audio.shape[1]:] = audio
				audio = _audio
			audiofeatures = torchaudio.transforms.MelSpectrogram(sample_rate=44100, win_length=882, hop_length=441, n_mels=64,
												   n_fft=1024, window_fn=torch.hann_window)(audio)
			#waveform, sr = torchaudio.load(audioPath, frame_offset=int(subseq_len*i), num_frames=int(subseq_len))

			max_spec_shape.append(audiofeatures.shape[2])
			waves.append(wave)
			#if specgram.shape[2] > 851:
			#	_audio_features = torch.zeros(1, 64, 851)
			#	_audio_features = specgram[:, :, -851:]
			#	audiofeatures = _audio_features
			#elif specgram.shape[2] < 851:
			#	_audio_features = torch.zeros(1, 64, 851)
			#	_audio_features[:, :, -specgram.shape[2]:] = specgram
			#	audiofeatures = _audio_features
			#else:
			#	audiofeatures = specgram
			#if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
			#	waveform, sr = torchaudio.load(audioPath)
			#else:
			#	waveform, sr = torchaudio.load(audioPath)
			#specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, win_length=400, hop_length=160, n_mels=128, n_fft=1024, normalized=True)(audio)
			#specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=640, hop_length=640, n_mels=128, n_fft=1024, normalized=True)(waveform)
			audio_feature = audio_spec_transform(audiofeatures)

			#plot_spectrogram(
    		#		audiofeatures[0], id, title="MelSpectrogram - torchaudio", ylabel='mel freq')

			# my transform
			#tensor = specgram.numpy()
			#res = np.where(tensor == 0, 1E-19 , tensor)
			#spectre = torch.from_numpy(res)

			#mellog_spc = spectre.log2()[0,:,:]#.numpy()
			#mean = mellog_spc.mean()
			#std = mellog_spc.std()
			#spec_norm = (mellog_spc - mean) / (std + 1e-11)
			#spec_min, spec_max = spec_norm.min(), spec_norm.max()
			#spec_scaled = (spec_norm/spec_max)*2 - 1
			spectrograms.append(audio_feature)
		spec_dim = max(max_spec_shape)
		audio_features = torch.zeros(len(max_spec_shape), 1, 64, spec_dim)
		for batch_idx, spectrogram in enumerate(spectrograms):
			if spectrogram.shape[2] < spec_dim:
				#print(batch_idx)
				#_audio_features = torch.zeros(1, 64, spec_dim)
				audio_features[batch_idx, :, :, -spectrogram.shape[2]:] = spectrogram
				#_audio_features[:, :, -spectrogram.shape[2]:] = spectrogram
				#audiofeatures = _audio_features
			else:
				audio_features[batch_idx, :,:, :] = spectrogram
		#melspecs_scaled = torch.stack(audio_features)

		#torch.cuda.synchronize()
		#t12 = time.time()
		return audio_features, waves # melspecs_scaled
