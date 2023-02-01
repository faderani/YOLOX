#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os.path

import cv2
import numpy as np
from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-p", "--predictions", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default=None, help="output directory to save the images")
    parser.add_argument("-c", "--confidence", type=str, default=None)
    return parser

def parse_predictions(txt_file_path):


    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        preds = []
        for l in lines:
            arr = l.split(" ")
            preds.append([arr[0], float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5])])


    return preds

def parse_predictions_dict(txt_file_path, conf=0.5):


    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        preds = {}
        for l in lines:
            arr = l.split(" ")
            if float(arr[1]) > conf:
                preds[arr[0]] = [float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5])]


    return preds

def main(txt_file_path, output_dir, confidence):
    preds = parse_predictions(txt_file_path)

    for path, conf, x1,y1,x2,y2 in tqdm(preds):
        full_path = f'{output_dir}/{path}.jpg'
        if os.path.exists(full_path) == False:
            #full_path = f'/ssd2/datasets/VOC/VOCdevkit/VOC2007/JPEGImages/{path}.jpg'
            full_path = f'/ssd2/datasets/EAT_hand_obj/VOCdevkit/VOC2007/JPEGImages/{path}.jpg'
        img = cv2.imread(full_path)
        img = cv2.resize(img, (480,640), interpolation = cv2.INTER_AREA)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        out_path = f'{output_dir}/{path}.jpg'
        cv2.imwrite(out_path, img)

def intersect_hand_obj(hand_txt_file_path, obj_txt_file_path, output_dir):

    hand_preds = parse_predictions_dict(hand_txt_file_path)
    obj_preds = parse_predictions_dict(obj_txt_file_path)


    for path, (conf, x1,y1,x2,y2) in hand_preds.items():
        if path in obj_preds:
            full_path = f'/ssd2/datasets/EAT_hand_obj/VOCdevkit/VOC2007/P44/JPEGImages/{path}.jpg'
            img = cv2.imread(full_path)
            img = cv2.resize(img, (480, 640), interpolation=cv2.INTER_AREA)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            conf, x1, y1, x2, y2 = obj_preds[path]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            out_path = f'{output_dir}/{path}.jpg'
            cv2.imwrite(out_path, img)











if __name__ == "__main__":

    #args = make_parser().parse_args()
    # hand_txt_file_path = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/out_of_the_box/comp4_det_test_hand.txt"
    # obj_txt_file_path = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/out_of_the_box/comp4_det_test_targetobject.txt"

    # hand_txt_file_path = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/finetune/comp4_det_test_hand.txt"
    # obj_txt_file_path = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/finetune/comp4_det_test_targetobject.txt"

    hand_txt_file_path = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/aug/P44/comp4_det_test_hand.txt"
    obj_txt_file_path = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/aug/P44/comp4_det_test_targetobject.txt"

    output_dir = "/ssd2/datasets/EAT_hand_obj/VOCdevkit/results/VOC2007/Main/aug/P44/hand_vis"

    # main(txt_file_path=hand_txt_file_path,
    #      output_dir=output_dir,
    #      confidence=0.001)

    intersect_hand_obj(hand_txt_file_path, obj_txt_file_path, output_dir)