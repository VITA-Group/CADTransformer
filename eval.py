import os
from tqdm import tqdm
from utils.utils_model import get_pred_instance
from config import anno_config
import torch
from pdb import set_trace as st

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1

def do_eval(model, loaders, logger, cfg):
    logger.info(f'> Conducting do_eval')
    with torch.no_grad():
        model = model.eval()
        anno_list = anno_config.AnnoList().anno_list_all_reverse
        class_num = len(anno_list)
        cnt_prd, cnt_gt, cnt_tp = \
            [torch.Tensor([0 for x in range(0, class_num+1)]).cuda() for _ in range(3)]
        valid_list = []
        with tqdm(loaders, total=len(loaders), smoothing=0.9) as _tqdm:
            for i, (image, xy, target, rgb_info, nns, offset_gt, inst_gt, index, basename) in enumerate(_tqdm):
                seg_pred = model(image, xy, rgb_info, nns)
                seg_pred = seg_pred.contiguous().view(-1, cfg.num_class+1)
                index = index.contiguous().view(-1).cpu().numpy()
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]
                # Squeeze
                xy = xy.squeeze(0)

                for prd, gt in zip(pred_choice, target):
                    cnt_prd[prd] += 1
                    cnt_gt[gt] += 1
                    if prd == gt:
                        cnt_tp[gt] += 1
            # Accumulating
            cls_id = 0
            precision = cnt_tp[cls_id]/(cnt_prd[cls_id]+1e-4)
            recall = cnt_tp[cls_id]/(cnt_gt[cls_id]+1e-4)
            f1 = (2*precision*recall) / (precision + recall+1e-4)
            logger.info("ID:[{:3s}], CLASS:[{:20s}], Pred Num: [{:0>7}], GT Num: [{:0>7}], F1:[{:.2%}], Precision:[{:.2%}], Recall:[{:.2%}]".format(\
                str(cls_id), "BG", cnt_prd[cls_id], cnt_gt[cls_id], f1, precision, recall))
            for cls_id in range(1, class_num):
                precision = cnt_tp[cls_id]/(cnt_prd[cls_id]+1e-4)
                recall = cnt_tp[cls_id]/(cnt_gt[cls_id]+1e-4)
                f1 = (2*precision*recall) / (precision + recall+1e-4)
                logger.info("ID:[{:3s}], CLASS:[{:20s}], Pred Num: [{:0>7}], GT Num: [{:0>7}], F1:[{:.2%}], Precision:[{:.2%}], Recall:[{:.2%}]".format(\
                    str(cls_id), anno_list[cls_id], cnt_prd[cls_id], cnt_gt[cls_id], f1, precision, recall))
            tp = sum(cnt_tp[1:])
            gt = sum(cnt_gt[1:])
            pred = sum(cnt_prd[1:])
            precision = tp / pred
            recall = tp / gt
            f1 = (2*precision*recall) / (precision + recall+1e-4)
            logger.info(f'Total FG Precision:{precision}, Recall:{recall}, F1:{f1}')
    return f1


def get_eval_criteria(epoch):
    eval = False
    if epoch < 50:
        if epoch % 5 == 0:
            eval = True
    if 50 < epoch < 1e5:
        if epoch % 5 == 0:
            eval = True
    if epoch == 0 or epoch == 1:
        eval = True
    return eval