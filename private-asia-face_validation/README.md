# Find optimal threshold and perform validation

- Model: [IR-50](https://drive.google.com/drive/folders/11TI4Gs_lO-fbts7cgWNqvVfm9nps2msE), trained on [ZhaoJ9014](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)'s private asia face dataset.

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

| Dataset | Accuracy | Optimal threshold |
| ------- | -------- | ----------------- |
| LFW     | 0.994    | 0.3525            |
| CALFW   | 0.939    | 0.3375            |
| CPLFW   | 0.897    | 0.375             |

- Visualize (red point is the optimal threshold in which we have the highest accuracy):

  - LFW:

  <img src="https://github.com/TuanMinhLe/Miscellaneous/blob/master/private-asia-face_validation/images/LFW_acc.png" width="500px"/>
  <img src="https://github.com/TuanMinhLe/Miscellaneous/blob/master/private-asia-face_validation/images/LFW_roc.png" width="500px"/>

  Range of threshold where accuracy >= 0.97: [0.2175, 0.4175].

  - CALFW:

  <img src="https://github.com/TuanMinhLe/Miscellaneous/blob/master/private-asia-face_validation/images/CALFW_acc.png" width="500px"/>
  <img src="https://github.com/TuanMinhLe/Miscellaneous/blob/master/private-asia-face_validation/images/CALFW_roc.png" width="500px"/>

  Range of threshold where accuracy >= 0.91: [0.28, 0.41].

  - CPLFW:

  <img src="https://github.com/TuanMinhLe/Miscellaneous/blob/master/private-asia-face_validation/images/CPLFW_acc.png" width="500px"/>
  <img src="https://github.com/TuanMinhLe/Miscellaneous/blob/master/private-asia-face_validation/images/CPLFW_roc.png" width="500px"/>

  Range of threshold where accuracy >= 0.87: [0.315, 0.43].

- Practical usage:

```
In real face-recognition problems, we should pick threshold as the mean of these numbers:
  - Optimal value: 0.355.
  - Range for adaptive selection: [0.271, 0.42].
```
