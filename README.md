# ECG UROP Code

<img width="790" alt="image" src="https://github.com/user-attachments/assets/49763466-e507-437e-93cc-afac62169cb1">

The project uses the PTBXL dataset, which is a large, publicly available electrocardiography dataset. You can untar the WFDB_PTBXL.tar.gz file to obtain the dataset. The data is stored in the `WFDB_PTBXL/` directory (which is gitignored).

## Model Architecture

This project's model architecture is adapted from the DSAIL_SNU2 team's code presented in the paper "Multi-Label Classification of 12-lead ECGs Using Deep Convolutional Neural Networks" (https://physionet.org/files/challenge-2021/1.0.2/papers/CinC2021-080.pdf). The DSAIL_SNU2 team's approach achieved high performance in the PhysioNet/Computing in Cardiology Challenge 2021, which focused on multiple cardiac abnormality classification using 12-lead ECG recordings.

From this foundation, we modify the architecture for our purposes by incorporating Knowledge Distillation (KD) loss between a 12-lead model (teacher) and a single-lead model (student). This approach aims to transfer the knowledge from the more comprehensive 12-lead model to the simpler single-lead model.

## Model Training

You can see how training works with the `train_model.sh` script. This script runs multiple training sessions with different configurations:

- Training on all leads
- Training on individual leads (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
- Training with and without KL divergence

The `train_model.py` script is called with various arguments to control the training process:

1. Data directory (e.g., WFDB_PTBXL)
2. Model output directory (e.g., new_model)
3. Random seed (e.g., 42)
4. Alpha value for KL divergence (e.g., 0 or 0.05) (0 for not using KD loss)
5. Number of KL layers (e.g., 0 - 4) (We found 2 to be good back in the day)
6. Split file path (e.g., splits/split_0.json)
7. Lead index (optional, 0-11 for specific lead, omit for all leads`)

The KL divergence is used as a regularization technique, which can help in preventing overfitting and improving generalization.

## Project Structure

- `train_model.py`: Main script for training the model
- `train_model.sh`: Shell script to run multiple training configurations
- `splits/`: Directory containing data split information
- `new_model/`: Directory where trained models are saved (gitignored)

## Running the Project


To reproduce the results from our previous experiments, follow these steps:

1. Train a 12-lead model:
   ```
   python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json
   ```
   This command trains the 12-lead model (teacher) on split 0.

2. Train a single-lead model without Knowledge Distillation:
   ```
   python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 0
   ```
   This trains a model using only lead I (index 0) without KD.

3. Train a single-lead model with Knowledge Distillation:
   ```
   python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 0
   ```
   This trains a model using lead I with KD, using an alpha value of 0.05 and applying KD on the last 2 ResBlock layers.


## Analyzing Results

To compare the results of different training configurations and evaluate the effectiveness of Knowledge Distillation (KD), you can follow these steps:

1. Run multiple training sessions:
   - Train models for multiple splits (e.g., split_0, split_1, split_2, etc.)
   - For each split, train models with and without KD for each lead

2. Compare performance using the logs:
   After training, you can find the logs in the `logs/` directory. Each training session will generate a log file containing performance metrics.

You can modify the lead index (last argument) to experiment with different leads. The results will be saved in the `new_model/` directory.
Note: You Need to use the same seed and split for the teacher and student, but this can be changed in `team_code.py` (Sorry, this is logic is poorly coded)


## Knowledge Distillation and LAYER_KL_WEIGHTS

In the knowledge distillation process, the `LAYER_KL_WEIGHTS` array determines the amount of KD to apply based on the layer (it's multiplier on top of the Alpha value). This array contains weights that control the contribution of each layer to the overall distillation loss. The weights are typically set to give more importance to the later layers of the network, as these layers often capture more abstract and task-specific features. You may want to refactor this to be controlled in `team_code.py` and try to mess with these values.



