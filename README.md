<<<<<<< HEAD
# CAPrompt
Class-aware Cross-attention Helps Prompt Learning with Limited Samples
=======

# Class-aware Cross-attention Helps Prompt Learning with Limited Samples


# Install


```
# Create a conda environment
conda create -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (version >= 1.8.1) and torchvision
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```


# Datsets
Please follow the CoOp instructions: [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).


# Run
First, you need set the dataset path in ./scripts/***.sh.
Then, navigate to the scripts folder:

```bash
cd ./scripts
```
## Base-to-New Generalization
Run the following command, where gpuid specifies the GPU ID you want to use.
```bash

bash base2new.sh gpuid
```

## Domain Generalization & Cross-dataset Evaluation
For domain generalization and cross-dataset evaluation, we first train on ImageNet with multiple GPUs, as shown below:
```bash
cd ./scripts
bash xd_train.sh gpuid1,gpuid2
```
Next, we perform Domain Generalization and Cross-dataset evaluation on the new dataset.
```bash
bash xd_test_cross.sh gpuid1,gpuid2
bash xd_test_dg.sh gpuid1,gpuid2
```



##  Few-shot  Evaluation
For few-shot evaluation, run the following command.  
The first two arguments specify the GPU IDs, and the third argument sets the number of shots.
```bash
bash few-shot.sh gpuid1,gpuid2 shots
```
>>>>>>> ce7ba0a (First commit)
