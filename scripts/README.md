#### **Step 0: Build0S3**
```bash
python scripts/TestBuild0S3.py
```

Process the dataset to `spacing (0.3, 0.3, 0.3)`, set `pulp_id = teeth_id+32`

#### **Step 1: Train Labeled**

```bash
python scripts/TestBuild611.py
python scripts/TestBuild711.py
python scripts/TestBuildWater.py
```

**DatasetX11_MICCAILabeled**
`1-32: tooth`, `33-64: tooth`
**Dataset621_MICCAILabeled**
`1-32: tooth`
**Dataset716_MICCAILabeled**
`1: tooth, 2: pulp`

```bash
nnUNetv2_extract_fingerprint -d 621
nnUNetv2_plan_experiment -d 621
nnUNetv2_preprocess -d 621 -c 3d_new
nnUNetv2_train 621 3d_new 0 --c -tr nnUNetTrainer_250epochs_NoMirroring -num_gpus 2

nnUNetv2_extract_fingerprint -d 716
nnUNetv2_plan_experiment -d 716
nnUNetv2_preprocess -d 716 -c 3d_new
nnUNetv2_train 716 3d_new 0 -tr nnUNetTrainer_250epochs
```


#### **Step 1: Train Unlabeled**
```bash
python scripts/TestBuild811Fake.py
python scripts/TestBuild821.py
python scripts/TestBuildWater.py
```

**Dataset821_MICCAIUnlabeled**
`1-32: tooth`
**Dataset916_MICCAIUnlabeled**
`1: tooth, 2: pulp`

```bash
nnUNetv2_extract_fingerprint -d 821
nnUNetv2_plan_experiment -d 821
nnUNetv2_preprocess -d 821 -c 3d_new
nnUNetv2_train 821 3d_new 0 -tr nnUNetTrainer_DASegOrd0_NoMirroring -num_gpus 2

nnUNetv2_extract_fingerprint -d 916
nnUNetv2_plan_experiment -d 916
nnUNetv2_preprocess -d 916 -c 3d_new
nnUNetv2_train 916 3d_new 0 -tr nnUNetTrainer
```

#### **Step 2: Train Unlabeled Spacing 1.0**
```bash
python scripts/TestBuild1011Fake.py
python scripts/TestBuild1211S1.py
python scripts/TestBuild1021.py
```
**Dataset1221_MICCAIUnlabeled**
`1-32: tooth, spacing(1.0, 1.0, 1.0)`

```bash
nnUNetv2_extract_fingerprint -d 1121
nnUNetv2_plan_experiment -d 1121
nnUNetv2_preprocess -d 1121 -c 3d_new
nnUNetv2_train 1121 3d_new 0 --c -tr nnUNetTrainer_DASegOrd0_NoMirroring
```
