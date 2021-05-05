import torch
import torchvision.transforms as transforms

import bcolz
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import io
import os

from optimize_threshold import evaluate


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, "lfw_align_112/lfw")
    calfw, calfw_issame = get_val_pair(data_path, "calfw_align_112/calfw")
    cplfw, cplfw_issame = get_val_pair(data_path, "cplfw_align_112/cplfw")

    return [[lfw, lfw_issame], [calfw, calfw_issame], [cplfw, cplfw_issame]]


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode="r")
    issame = np.load("{}/{}_list.npy".format(path, name))

    return carray, issame


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5


hflip = transforms.Compose(
    [
        de_preprocess,
        transforms.ToPILImage(),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


ccrop = transforms.Compose(
    [
        de_preprocess,
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.CenterCrop([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def gen_plot(fpr, tpr):
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    plt.close()

    return buf


def perform_val(
    device, embedding_size, batch_size, backbone, carray, issame, nrof_folds
):

    backbone = backbone.to(device)
    backbone.eval()

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx : idx + batch_size][:, [2, 1, 0], :, :])

            ccropped = ccrop_batch(batch)
            fliped = hflip_batch(ccropped)
            emb_batch = (
                backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
            )
            embeddings[idx : idx + batch_size] = l2_norm(emb_batch)

            idx += batch_size

        if idx < len(carray):
            batch = torch.tensor(carray[idx:])

            ccropped = ccrop_batch(batch)
            fliped = hflip_batch(ccropped)
            emb_batch = (
                backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
            )
            embeddings[idx:] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, accuracy, best_threshold, roc_curve):
    writer.add_scalar("{}_Accuracy".format(db_name), accuracy)
    writer.add_scalar("{}_Best_Threshold".format(db_name), best_threshold)
    writer.add_image("{}_ROC_Curve".format(db_name), roc_curve)

>