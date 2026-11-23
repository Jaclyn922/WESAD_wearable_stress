# Wrist-Based Stress Detection in Unseen Subjects

This project investigates whether wrist-worn physiological signals can distinguish stress from baseline in unseen participants, whether personal baseline calibration improves performance, and whether models trained on WESAD generalize to an independent dataset.

## Notebooks

| File | Description |
|---|---|
| `file1.ipynb` | WESAD preprocessing, feature extraction, personal baseline calibration, leave-one-subject-out evaluation, signal ablation, and sensitivity analyses. |
| `JX_externalvalidation.ipynb` | Training models on WESAD and evaluating them on an independent PhysioNet dataset, including comparisons of calibration, ACC removal, and non-stress definitions. |

## Datasets

**WESAD**
- 15 participants.
- Baseline versus stress induced through public speaking and mental arithmetic.
- Used for model development and leave-one-subject-out (LOSO) cross-validation.
- [Dataset information](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)

**Wearable Device Dataset from Induced Stress and Structured Exercise Sessions**
- Independent dataset hosted on PhysioNet.
- This project uses its `STRESS` recordings, including initial baseline, Stroop/TMCT tasks, and rest periods.
- 33 participants retained after recording exclusions; 29 included in the primary initial-baseline-versus-stress evaluation.
- [Dataset information and download](https://physionet.org/content/wearable-device-dataset/1.0.1/)

Both datasets include Empatica E4 wrist recordings: BVP at 64 Hz, EDA and skin temperature at 4 Hz, and three-axis acceleration at 32 Hz.

Raw datasets are not included in this repository. Download them separately and follow their usage and citation requirements.

## Methods

- Divide recordings into non-overlapping 60-second windows within condition boundaries.
- Extract 28 features from BVP, EDA, temperature, and acceleration.
- Compare uncalibrated features with features centered on each participant’s first two baseline windows.
- Exclude calibration windows from classifier training and evaluation in both settings.
- Evaluate decision tree (DT), linear discriminant analysis (LDA), and random forest (RF) classifiers.

For WESAD, LOSO evaluation trains on 14 participants and tests on the remaining participant. Imputation, standardization, and classifier fitting use only the training participants.

For external validation, models are fitted on WESAD and applied to the external dataset without classifier retraining or tuning. The calibrated setting uses each external participant’s initial resting baseline as a personal reference.

Balanced accuracy is the primary metric. Stress F1 and recall are also reported. Summary metrics are averaged across participants.

## Main Results

### WESAD: Mean Balanced Accuracy

| Model | Uncalibrated | Two-minute calibration |
|---|---:|---:|
| DT | 0.820 | 0.935 |
| LDA | 0.917 | 0.913 |
| RF | 0.886 | 0.933 |

Calibration improved average performance for DT and RF, but benefits varied across participants. Signal ablation showed that contributions depended on the classifier: removing EDA caused the largest average decrease for DT and RF, while removing BVP caused the largest decrease for LDA.

### External Validation: Mean Balanced Accuracy

Primary evaluation: initial baseline versus stress, using all 28 features and the same 29 participants.

| Model | Uncalibrated | Two-minute calibration |
|---|---:|---:|
| DT | 0.586 | 0.707 |
| LDA | 0.601 | 0.690 |
| RF | 0.579 | 0.657 |

Calibration improved mean balanced accuracy for all three models, but none of these external gains was statistically significant after Holm correction. Performance was substantially lower than in WESAD LOSO evaluation.

Supplementary analyses include recovery rest in the non-stress class. Under these definitions, calibrated models performed worse on average than uncalibrated models, highlighting sensitivity to the evaluation protocol.

## Running the Notebooks

Install the required Python packages:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn neurokit2 joblib jupyterlab
```

1. Download and extract both datasets.
2. Update the data and output paths in each notebook to match your local folders.
3. Run `file1.ipynb` from top to bottom to generate the WESAD feature tables and results.
4. Configure `JX_externalvalidation.ipynb` to use those exported WESAD tables and the external dataset’s `Wearable_Dataset/STRESS` folder.
5. Run the external validation notebook from top to bottom.

The notebooks save generated tables and figures to their configured output directories.

## Limitations

- The number of participants is small, and windows from the same participant are not independent observations.
- Stress tasks may differ from baseline in movement, posture, and speaking. Removing ACC features does not eliminate movement-related effects in physiological signals.
- BVP peak detection and SCR extraction have quality limitations; successfully computed features are not necessarily physiologically accurate.
- The primary external evaluation contains only 30 baseline test windows across 29 participants.
- The external Version 2 baseline is assumed to be the 180 seconds before the first event tag. This boundary remains unverified, so external findings should be interpreted as exploratory.
- Both datasets use controlled protocols; these results do not establish reliable everyday stress detection.

## Future Work

Validate event boundaries, improve signal-quality checks, and evaluate larger and more diverse populations. Experiments that vary movement and psychological stress independently would help distinguish stress-related information from task-related activity.