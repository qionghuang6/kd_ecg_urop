# Training on all leads for split 0
python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json

# Training on lead I, with and without KL divergence
python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 0 # I No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 0 # I With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 1 # II No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 1 # II With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 2 # III No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 2 # III With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 3 # aVR No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 3 # aVR With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 4 # aVL No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 4 # aVL With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 5 # aVF No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 5 # aVF With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 6 # V1 No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 6 # V1 With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 7 # V2 No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 7 # V2 With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 8 # V3 No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 8 # V3 With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 9 # V4 No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 9 # V4 With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 10 # V5 No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 10 # V5 With KL

python train_model.py WFDB_PTBXL new_model 42 0 0 splits/split_0.json 11 # V6 No KL
python train_model.py WFDB_PTBXL new_model 42 0.05 2 splits/split_0.json 11 # V6 With KL

# $1: Data directory (e.g., WFDB_PTBXL)
# $2: Model output directory (e.g., new_model)
# $3: Random seed (e.g., 42)
# $4: Alpha value for KL divergence (e.g., 0 or 0.05)
# $5: Number of KL layers (e.g., 0 - 4)
# $6: Split file path (e.g., splits/split_0.json)
# $7: Lead index (optional, 0-11 for specific lead, omit for all leads)

# NOTE: Look at LAYER_KL_WEIGHTS in train.py to see how the KL divergence weights are set by ResBlock