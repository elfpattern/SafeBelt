#!/usr/bin/env mdl
import json
import sys
from tqdm import tqdm
import argparse
max_iou = 0.5
min_iou = 0.3
max_ioa = 0.5
odgt_nid = {}
oddet_nid = {}
ret_nid = {}

count_pos = 0
count_neg = 0
def output(ret_nid, final_odgt):
    with open(final_odgt, 'w') as f:
        global f_oddet
        for file in tqdm(f_oddet):
            nid = eval(file)['noriID']
            if nid in ret_nid.keys():
                odf = {}
                odf['dtboxes'] = ret_nid[nid]
                odf['noriID'] = nid
                odf['height'] =eval(file)['height']
                odf['width'] = eval(file)['width']
                odf['ID'] = nid
                odf['noriroot'] = eval(file)['noriroot']
                json.dump(odf, f)
                print('', file=f)


def calcIOU(dtbox, gtbox):
    one_x,one_y,one_w,one_h = int(dtbox['box'][0]),int(dtbox['box'][1]), int(dtbox['box'][2]), int(dtbox['box'][3])
    two_x,two_y,two_w,two_h = int(gtbox['box'][0]),int(gtbox['box'][1]), int(gtbox['box'][2]), int(gtbox['box'][3])
    if((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        IOU = inter_square / union_square * 1.0
    else:
        IOU = 0.0
    return IOU

def calcIOA(dtbox, gtbox):
    one_x,one_y,one_w,one_h = int(dtbox['box'][0]),int(dtbox['box'][1]), int(dtbox['box'][2]), int(dtbox['box'][3])
    two_x,two_y,two_w,two_h = int(gtbox['box'][0]),int(gtbox['box'][1]), int(gtbox['box'][2]), int(gtbox['box'][3])
    if((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        dtbox_square = (one_w * one_h)

        IOA = inter_square / dtbox_square * 1.0
    else:
        IOA = 0.0

    return IOA

def add_box(nid, dtbox, tag):
    global ret_nid
    if nid not in ret_nid:
        ret_nid[nid] = []
    dtbox = {'box' : dtbox['box'], 'tag' : tag, 'extra' : {'ignore' : 0}}
    ret_nid[nid].append(dtbox)

def make_odgt(odgt, oddet, get_odgt):
    for k in tqdm(oddet.keys()):
        if k in odgt.keys():
            for dtbox in oddet[k]:
                if dtbox['score'] < 0.3:
                    continue
                max_IoU = 0.0
                max_IoA = 0.0
                for gtbox in odgt[k]:
                    x1,y1,w1,h1 = int(gtbox['box'][0]),int(gtbox['box'][1]), int(gtbox['box'][2]), int(gtbox['box'][3])
                    if gtbox['extra']['ignore'] == 1:
                        ioa = calcIOA(dtbox, gtbox)
                        max_IoA = max(max_IoA, ioa)
                    else:
                        iou = calcIOU(dtbox,gtbox)
                        max_IoU = max(max_IoU, iou)
                if max_IoA > max_ioa :
                    continue
                if max_IoU > max_iou:
                    if dtbox['tag'] == 'truck':
                        global count_pos
                        count_pos += 1
                    add_box(k, dtbox, '__TP__')
                if max_IoU < min_iou:
                    if dtbox['tag'] == 'truck':
                        global count_neg
                        count_neg += 1
                    add_box(k, dtbox, '__FP__')
        else:
            print('it is not match')

    print('pos_truck:', count_pos)
    print('neg_truck:', count_neg)
    output(ret_nid, get_odgt)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt','--gtbox')
    parser.add_argument('-dt', '--detbox')
    parser.add_argument('-t','--train_odgt',help='get the data to train net')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    f_odgt = open(args.gtbox)
    f_odgt = f_odgt.readlines()
    for file in tqdm(f_odgt):
        file = json.loads(file)
        odgt_nid[file['noriID']] = file['gtboxes']

    f_oddet = open(args.detbox)
    f_oddet = f_oddet.readlines()
    for file in tqdm(f_oddet):
        file = json.loads(file)
        oddet_nid[file['noriID']] = file['dtboxes']
    make_odgt(odgt_nid, oddet_nid, args.train_odgt)


