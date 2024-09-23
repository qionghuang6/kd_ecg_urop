'''
Generate sample data for the curve generation.
The sample data is an aneurysm detection scenario.
'''

# %%
import pandas as pd
import numpy as np
import scipy.stats

import generate_sample_data_config as config

# %%
# random generator
rng = np.random.default_rng(0)

# %%
# First generate the patient list
df_patients = pd.DataFrame({
    'MRN': ['IA{0:05d}'.format(i + 1) for i in range(config.npatients)],
    'Dataset': 'test',
    'Site': rng.choice(['Site{0}'.format(i + 1) for i in range(config.nsites)], config.npatients, True)
})


# %%
# Then generate the aneurysm for each patient
def generate_bbox(img_size, size_min, size_max, pixel_size, rng=None):
    rng = np.random.default_rng(rng)

    # Center
    center = rng.uniform(img_size)
    size_in_mm = rng.uniform(size_min, size_max)

    bbox = list(center) + [size_in_mm / pixel_size]

    return bbox, size_in_mm


def generate_label_bboxes(
    img_size, size_min, size_max, pixel_size,
    lesion_counts, lesion_counts_prob, locations,
    rng=None
):
    rng = np.random.default_rng(rng)

    # number of lesions
    nlesion = rng.choice(lesion_counts, p=lesion_counts_prob)
    if nlesion == 0:
        return pd.DataFrame(
            columns=['bbox', 'prob', 'Tag', 'DatasetTag', 'AneurysmSize', 'AneurysmID', 'Position']
        )

    df = []
    for i in range(nlesion):
        bbox, size_in_mm = generate_bbox(
            img_size, size_min, size_max, pixel_size, rng
        )
        df.append({
            'bbox': np.array([1] + bbox),
            'prob': 1,
            'Tag': 'annotation',
            'DatasetTag': 'test',
            'AneurysmSize': size_in_mm,
            'AneurysmID': i,
            'Position': rng.choice(locations)
        })
    return pd.DataFrame(df)


def generate_tp_bbox(
    label_bbox, center_offset_max, rel_size_min, rel_size_max, pixel_size,
    tp_prob_mean, tp_prob_std, tp_per_lesion, tp_per_lesion_prob,
    rng=None
):
    rng = np.random.default_rng(rng)

    ntp = rng.choice(tp_per_lesion, p=tp_per_lesion_prob)
    if ntp == 0:
        return pd.DataFrame(
            columns=['bbox', 'prob', 'Tag', 'DatasetTag', 'AneurysmSize', 'AneurysmID', 'Position']
        )

    df = []
    for i in range(ntp):
        bbox, size_in_mm = generate_bbox(
            np.array([label_bbox[4]] * 3) * center_offset_max,  # relative position to the label center
            label_bbox[4] * pixel_size * rel_size_min,
            label_bbox[4] * pixel_size * rel_size_max,
            pixel_size,
            rng
        )
        # offset the bbox so it's centered on the label center
        bbox[:3] = bbox[:3] - np.array([label_bbox[4]] * 3) * center_offset_max / 2 + label_bbox[1:4]
        a = (0.001 - tp_prob_mean) / tp_prob_std
        b = (0.999 - tp_prob_mean) / tp_prob_std
        prob = scipy.stats.truncnorm.rvs(a, b, tp_prob_mean, tp_prob_std)
        df.append({
            'bbox': np.array([prob] + bbox),
            'prob': prob,
            'Tag': 'prediction',
            'DatasetTag': 'test',
            'AneurysmSize': size_in_mm,
            'AneurysmID': None,
            'Position': None
        })
    return pd.DataFrame(df)


def generate_fp_bbox(
    img_size, size_min, size_max, pixel_size,
    fp_prob_mean, fp_prob_std, max_fp_per_patient,
    rng=None
):
    rng = np.random.default_rng(rng)

    nfp = rng.integers(max_fp_per_patient + 1)
    if nfp == 0:
        return pd.DataFrame(
            columns=['bbox', 'prob', 'Tag', 'DatasetTag', 'AneurysmSize', 'AneurysmID', 'Position']
        )

    df = []
    for i in range(nfp):
        bbox, size_in_mm = generate_bbox(
            img_size, size_min, size_max, pixel_size, rng
        )
        a = (0.001 - fp_prob_mean) / fp_prob_std
        b = (0.999 - fp_prob_mean) / fp_prob_std
        prob = scipy.stats.truncnorm.rvs(a, b, fp_prob_mean, fp_prob_std)
        df.append({
            'bbox': np.array([prob] + bbox),
            'prob': prob,
            'Tag': 'prediction',
            'DatasetTag': 'test',
            'AneurysmSize': size_in_mm,
            'AneurysmID': None,
            'Position': None
        })
    return pd.DataFrame(df)


# %%
df_aneurysms = []
print(config.npatients)
for i in range(config.npatients):
    if (i + 1) % 100 == 0:
        print(i + 1, end=',', flush=True)

    pixel_size = rng.uniform(config.pixel_size_min, config.pixel_size_max)
    mrn = 'IA{0:05d}'.format(i + 1)

    # generate label
    df_label = generate_label_bboxes(
        config.img_size,
        config.lesion_size_min,
        config.lesion_size_max,
        pixel_size,
        config.lesion_counts,
        config.lesion_counts_prob,
        config.locations,
        rng
    )
    if len(df_label) > 0:
        df_label['MRN'] = mrn
        df_aneurysms.append(df_label)

    # generate TP
    df_tp = []
    for k, row in df_label.iterrows():
        df_tp.append(generate_tp_bbox(
            row['bbox'],
            config.center_offset_max,
            config.rel_size_min,
            config.rel_size_max,
            pixel_size,
            config.tp_prob_mean,
            config.tp_prob_std,
            config.tp_per_lesion,
            config.tp_per_lesion_prob,
            rng
        ))
    if len(df_tp) > 0:
        df_tp = pd.concat(df_tp)
        df_tp['MRN'] = mrn
        df_aneurysms.append(df_tp)

    # generate FP
    df_fp = generate_fp_bbox(
        config.img_size,
        config.lesion_size_min,
        config.lesion_size_max,
        pixel_size,
        config.fp_prob_mean,
        config.fp_prob_std,
        config.max_fp_per_patient,
        rng
    )
    if len(df_fp) > 0:
        df_fp['MRN'] = mrn
        df_aneurysms.append(df_fp)

df_aneurysms = pd.concat(df_aneurysms)

# %%
# save the results
df_patients.to_csv('./patients.csv', index=False)
df_aneurysms.to_csv('./aneurysms.csv', index=False)
