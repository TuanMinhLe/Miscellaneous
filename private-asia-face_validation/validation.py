import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from utils import get_val_data, perform_val, buffer_val
from model_irse import IR_50

"""
    Perform validation for IR-50 model (trained on ZhaoJ9014's private asia face dataset)
    on LFW, CALFW, CPLFW (all are aligned to size of 112x112)
"""

device = torch.device("cuda:0")
writer = SummaryWriter("buffer/log")
names = ["LFW_align_112", "CALFW_align_112", "CPLFW_align_112"]

val_data = get_val_data("./")

backbone = IR_50([112, 112])
backbone.load_state_dict(torch.load("backbone_ir50_asia.pth"))
backbone = backbone.to(device)

for item, name in zip(val_data, names):
    accuracy, best_threshold, roc_curve = perform_val(
        device, 512, 512, backbone, item[0], item[1], 10
    )

    buffer_val(writer, name, accuracy, best_threshold, roc_curve)

    print("=" * 60)
    print("Validation on {} dataset:".format(name))
    print("Accuracy (F1 score): {}".format(accuracy))
    print("Best threshold: {}".format(best_threshold))
    print("ROC curve: {}".format(roc_curve))

