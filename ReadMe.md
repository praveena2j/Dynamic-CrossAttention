### Description
This folder has the code for the paper "Cross-Attention is Not Always Needed: Dynamic Cross-Attention for Audio-Visual Dimensional Emotion Recognition" accepted to ICME2024. 

### Step 1 (Data Preprocessing)
```
Use the Datapreprocessing Code ("MySetup") to preprocess and extract the aligned segmented audio files and preprocess the labels.
```
### Step 2 (Pretraining the backbones)
```
In order to obtain the pretrained audio and visual features, we have used the TSAV setup (https://github.com/kuhnkeF/ABAW2020TNT). The pretrained model weights ("TSAV_Sub4_544k.pth.tar") are provided in the "PretrainedWeights" folder. 
```
### Step 3 (creating the conda environment)
```
conda env create -f environment.yml
```
### Step 4 (Run the code for Training)
```
A config file is provided, where we can pass the appropriate arguments and the paths for the data and labels. Please ensure to provide the correct paths in the configfile and use the default configeration for the hyper parameters. Run the shell script to start the training process. After training, the best model weights will be saved in "SavedWeights" folder.
The best model weights can be used for validation and testing.
```

### Code Structure
main.py  ---> takes the input arguments, loads the data, initializes the model, and pass it to train and validaion scripts.
train.py  ---> this script has the train function used to train the model with the Affwild2 dataset and returns the CCC accuracy of valence and arousal.
val.py  --->  this script has the validation function used to validate our model.
test.py  --> this script is used to load the trained model and generate the predictions of test data in txt format.
The scripts for models, data loading, loss and evalutaion scripts are provided in the respecitve folders.