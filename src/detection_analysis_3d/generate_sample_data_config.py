'''
Parameter configuration for sample data generation
'''

# %%
# parameters of the patients
# number of patients
npatients = 1000
# image size in (x, y, z), which gives the min and max value of the aneurysm centers
img_size = [512, 512, 512]
# number of sites performing the study
nsites = 2
# possible locations of the aneurysm
locations = ['ICA', 'PCOM', 'MCA']
# min and max size of lesion in mm
lesion_size_min = 3
lesion_size_max = 15
# possible number of lesions in each patient and their probability
lesion_counts = [0, 1, 2, 3, 4]
lesion_counts_prob = [0.5, 0.4, 0.05, 0.025, 0.025]
# min and max pixel size for each image
pixel_size_min = 0.3
pixel_size_max = 0.5

# parameters of the model
# True positives
# num of TP predictions per lesion and their probabilities
tp_per_lesion = [0, 1, 2, 3]
tp_per_lesion_prob = [0.05, 0.9, 0.04, 0.01]
# maximum offset of the center compared to the label
center_offset_max = 0.25
# min and max of the predicted size compared to the label
rel_size_min = 0.8
rel_size_max = 1.2
# probability distribution (gaussian)
tp_prob_mean = 0.9
tp_prob_std = 0.25
# False positives
max_fp_per_patient = 15
fp_prob_mean = 0.1
fp_prob_std = 0.25
