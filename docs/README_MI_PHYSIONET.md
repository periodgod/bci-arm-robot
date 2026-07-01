# PhysioNet Left/Right Motor-Imagery Benchmark

Source: [PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)

Downloaded data:

- Subject: `S001`
- Runs: `R04`, `R08`, `R12`
- Task: imagined left fist (`T1`) versus imagined right fist (`T2`)
- Sampling rate: `160 Hz`
- EEG channels: `64`
- Epoch: `0.5-3.5 s` after the cue
- Trials: `45` (`23 left`, `22 right`)

Converted project data:

- `data/processed/physionet_eegmmidb_s001_left_right_mi.npz`
- Shape: `(45, 64, 480)`
- Labels: `left`, `right`
- Unit: microvolts

## Validation

All scores use 5-fold stratified **trial-level** cross-validation. Windows from one trial never appear in both training and validation.

| Model | Accuracy | Balanced accuracy | Kappa |
|---|---:|---:|---:|
| Compact CSP-LDA, 21 motor channels | 86.67% | 86.76% | 0.734 |
| Existing FBCSP-LDA | 84.44% | 84.29% | 0.688 |
| Existing FBCSP-LogReg | 82.22% | 82.11% | 0.644 |
| Deployment CSP-LDA, device-channel intersection | 77.78% | 77.87% | 0.556 |
| EEGNet, device-channel intersection | 75.56% | 75.99% | 0.515 |
| ShallowConvNet, device-channel intersection | 75.56% | 75.79% | 0.513 |
| Log-covariance LogReg | 73.33% | 73.32% | 0.466 |
| Hemispheric filter-bank LDA | 68.89% | 69.07% | 0.380 |

The deep models are implemented and functional, but this single-subject set has only 45 trials. The deployable CSP-LDA model is therefore the current default. EEGNet and ShallowConvNet should be revisited after collecting subject-specific calibration data or pretraining on more subjects.

## Runtime fusion

EMG and EEG are deliberately separated:

| EMG axis selection | Left-hand MI | Right-hand MI |
|---|---|---|
| Left masseter | left | right |
| Right masseter | forward | backward |
| Both masseters | up | down |

This prevents the EEG classifier from learning facial EMG leakage as a shortcut. The saved deployment model is:

`models/mi_experiments/physionet_eegmmidb_s001_left_right_mi_deployment_csp_lda.joblib`

Reproduce the benchmark:

```powershell
& F:\anaconda\Scripts\conda.exe run -n bciwork_1 python tools\mi_physionet_benchmark.py --data-dir data\processed --raw-data-dir data\raw --subject 1 --runs 4 8 12 --folds 5 --deep-epochs 60 --no-download
```
