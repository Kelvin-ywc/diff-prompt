import copy
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data

from maskrcnn_benchmark import data
import maskrcnn_benchmark.utils.dist as dist
from maskrcnn_benchmark.layers.set_loss import generalized_box_iou

from .modulated_coco import ModulatedDataset
import logging

class RefExpDataset(ModulatedDataset):
    pass


class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, img_ids, k=(1, 5, -1), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        # self.img_ids = img_ids
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou
        self.catId2superCatName = {}
        self.superCatNames = []
        for key in self.refexp_gt.cats.keys():
            superCatName = self.refexp_gt.cats[key]["supercategory"]
            self.catId2superCatName[key] = superCatName
            if superCatName not in self.superCatNames:
                self.superCatNames.append(superCatName)

        self.logger = logging.getLogger("maskrcnn_benchmark.trainer")

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            dataset2score = {
                "refcoco": {k: 0.0 for k in self.k},
                "refcoco+": {k: 0.0 for k in self.k},
                "refcocog": {k: 0.0 for k in self.k},
            }
            dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}

            datasetBySuperCat2score = {
                superCatName: {k: 0.0 for k in self.k} for superCatName in self.superCatNames
            }
            datasetBySuperCat2count = {superCatName: 0.0 for superCatName in self.superCatNames}
            r_1_list = []
            for image_id in self.img_ids:
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                assert len(ann_ids) == 1
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                target = self.refexp_gt.loadAnns(ann_ids[0])
                superCatId = target[0]["category_id"]
                superCatName = self.catId2superCatName[superCatId]

                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                try:
                    sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                    sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                    target_bbox = target[0]["bbox"]
                    converted_bbox = [
                        target_bbox[0],
                        target_bbox[1],
                        target_bbox[2] + target_bbox[0],
                        target_bbox[3] + target_bbox[1],
                    ]
                    giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
                    for k in self.k:
                        if k == -1:
                            if max(giou) >= self.thresh_iou:
                                dataset2score[img_info["dataset_name"]][k] += 1.0
                                datasetBySuperCat2score[superCatName][k] += 1.0
                        elif max(giou[:k]) >= self.thresh_iou:
                            r_1_list.append(image_id)
                            dataset2score[img_info["dataset_name"]][k] += 1.0
                            datasetBySuperCat2score[superCatName][k] += 1.0
                    dataset2count[img_info["dataset_name"]] += 1.0
                    datasetBySuperCat2count[superCatName] += 1.0
                except:
                    dataset2count[img_info["dataset_name"]] += 1.0
                    datasetBySuperCat2count[superCatName] += 1.0

            with open('./save_res/r1.txt', 'w') as file:
                for item in r_1_list:
                    file.write(f"{item}\n")
            print('****************************************R1 Length:', len(r_1_list))
            self.logger.info(f"score: {datasetBySuperCat2score}, \n count: {datasetBySuperCat2count} \n")

            for key, value in dataset2score.items():
                for k in self.k:
                    try:
                        value[k] /= dataset2count[key]
                    except:
                        pass
            for key, value in datasetBySuperCat2score.items():
                for k in self.k:
                    try:
                        value[k] /= datasetBySuperCat2count[key]
                    except:
                        pass

            results = {}

            resultsBySuperCat = {}

            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                print(f" Dataset: {key} - Precision @ {self.k[0]}, {self.k[1]}, {self.k[2]}: {results[key]} \n")
                self.logger.info(f" Dataset: {key} - Precision @ {self.k[0]}, {self.k[1]}, {self.k[2]}: {results[key]} \n")
            

            for key, value in datasetBySuperCat2score.items():
                resultsBySuperCat[key] = sorted([v for k, v in value.items()])
                print(f" Super Category: {key} - Precision @ {self.k[0]}, {self.k[1]}, {self.k[2]}: {resultsBySuperCat[key]} \n")
                self.logger.info(f" Super Category: {key} - Precision @ {self.k[0]}, {self.k[1]}, {self.k[2]}: {resultsBySuperCat[key]} \n")
            return results
        return None
