import os
import shutil
import sys

Train_path = 'VA_Set/Train_Set'
Val_path = 'VA_Set/Val_Set'


dest_train_path = 'Fold5/Train_Set'
os.makedirs(dest_train_path, exist_ok=True)

dest_val_path = 'Fold5/Val_Set'
os.makedirs(dest_val_path, exist_ok=True)


list_files = os.listdir(Train_path)
val_list_files = os.listdir(Val_path)
folds = []
for i in range(5):
    if i == 4:
            folds.append(list_files[71*i:])
    else:
        folds.append(list_files[71*i:71*(i+1)])


for file in folds[4]: 
    shutil.copy(os.path.join(Train_path, file), os.path.join(dest_val_path, file))

for i in [0,1,2,3]:
    for file in folds[i]:
        shutil.copy(os.path.join(Train_path, file), os.path.join(dest_train_path, file))
    for valfile in val_list_files:
        shutil.copy(os.path.join(Val_path, valfile), os.path.join(dest_train_path, valfile))

print(len(folds[0]))
print(len(folds[1]))
print(len(folds[2]))
print(len(folds[3]))
print(len(folds[4]))
print(len(val_list_files))