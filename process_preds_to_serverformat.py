import sys
import os

format_file = 'CVPR_5th_ABAW_VA_test_set_sample.txt'
video_length = 0
results_dir = 'gating_results_14epoch'
results_output_dir = 'gating_results_bestVA_new_refined'
os.makedirs(results_output_dir, exist_ok=True)
new_video_length = 0
test_file_new = open('predictions_14epoch.txt', 'w')

test_file_new.write("image_location,valence,arousal")
test_file_new.write('\n')
count = 0
with open(format_file, 'r') as file:
	lines = list(file)[1:]
	for i in range(162):
		line = lines[new_video_length]
		find_str = os.path.split(os.path.splitext(line)[0])[0]
		for line in lines:			
			if find_str == os.path.splitext(line)[0][:-6]:
				new_video_length = new_video_length + 1
			else:
				break
		test_file = open(os.path.join(results_dir, find_str)+ '.txt', 'r')
		test_lines = list(test_file)[1:]
		if new_video_length != len(test_lines):
			count = count + 1
			
			#print("Template no:" +str(new_video_length))
			#print("results no:" + str(len(test_lines)))
			if new_video_length > len(test_lines):
				print(find_str)
				#print(test_lines[-1])
				imagepath = test_lines[-1].split(',')[0]
				val = test_lines[-1].split(',')[1]
				aro = test_lines[-1].split(',')[2]
				vid_name = os.path.normpath(imagepath).split(os.sep)[0]
				vidframe_id = int(os.path.normpath(imagepath).split(os.sep)[1][:-4])+1
				lastitem = ','.join([os.path.join(vid_name, str(vidframe_id).zfill(5) +'.jpg'), val, aro])
				#sys.exit()
				#test_lines.append(test_lines[-1])
				test_lines.append(lastitem)
			else:
				test_lines = test_lines[:new_video_length]
		if new_video_length != len(test_lines):
			print(find_str)
			print("not at all ok")
			sys.exit()

		#test_file_new = open(os.path.join(results_output_dir, find_str)+ '.txt', 'w')

		test_file_new.writelines(test_lines)
		lines = lines[new_video_length:]
		new_video_length = 0
print(count)
sys.exit()



