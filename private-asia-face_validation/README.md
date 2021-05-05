# Find optimal threshold and perform validation

- Model: IR-50, trained on [ZhaoJ9014](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)'s private asia face dataset.

- Datasets:

| Name  | Align   | Link                                                                           |
| ----- | ------- | ------------------------------------------------------------------------------ |
| LFW   | 112x112 | [link](https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT/view) |
| CALFW | 112x112 | [link](https://drive.google.com/file/d/1kpmcDeDmPqUcI5uX0MCBzpP_8oQVojzW/view) |
| CPLFW | 112x112 | [link](https://drive.google.com/file/d/14vPvDngGzsc94pQ4nRNfuBTxdv7YVn2Q/view) |

- Hardware: GeForce RTX 3060 (CUDA ID: "0").
- File structure:

```bash
├── model_irse.py
│
└── validation.py
    └── utils.py
        └── optimize_threshold.py
```

- Result:

| Dataset | F1 score | ROC curve |
| ------- | -------- | --------- |
| LFW     | 0.995    | 1.344     |
| CALFW   | 0.944    | 1.379     |
| CPLFW   | 0.896    | 1.504     |

- Visualize: 
<img src="https://github.com/TuanMinhLe/Miscellaneous/tree/master/private-asia-face_validation/images/LFW_ROC-curve.png" width="900px"/>
<img src="https://github.com/TuanMinhLe/Miscellaneous/tree/master/private-asia-face_validation/images/CALFW_ROC-curve.png" width="900px"/>
<img src="https://github.com/TuanMinhLe/Miscellaneous/tree/master/private-asia-face_validation/images/CPLFW_ROC-curve.png" width="900px"/>
