# Creates training_labels.csv and validation_labels.csv for use in tensorflow file

wnids_path = "tiny-imagenet-200/wnids.txt"
valannot_path = "tiny-imagenet-200/val/val_annotations.txt"
train_path = "tiny-imagenet-200/train/"
val_path = "tiny-imagenet-200/val/images/"

train_label_name = "training_labels.csv"
validation_label_name = "validation_labels.csv"

with open(wnids_path, 'r') as f:
	wnids_list = f.readlines()
	wnids_list = [x.strip() for x in wnids_list]

with open(train_label_name, 'w') as f:
	for i in range(len(wnids_list)):
		nid = wnids_list[i]
		for j in range(500):
			line = train_path + nid + "/images/" + nid + "_" + str(j) + ".JPEG," + str(i) + "\n"
			if (i == (len(wnids_list)-1) and j == 499):
				line = line[:-1]
			f.write(line)

with open(valannot_path, 'r') as f:
	valannot_list = f.readlines()
	filelist = []
	labellist = []
	for line in valannot_list:
		split = line.split("\t")
		filelist.append(split[0])
		labellist.append(split[1])

with open(validation_label_name, 'w') as f:
	for i in range(len(filelist)):
		for j in range(len(wnids_list)):
			if labellist[i] == wnids_list[j]:
				line = val_path + filelist[i] + "," + str(j) + "\n"
				if (i == len(filelist) - 1):
					line = line[:-1]
				f.write(line)

