import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from .BoundingBox import BoundingBox
from .BoundingBoxes import BoundingBoxes
from .utils import *

class Evaluator:
    cocoThresholds = [thresh / 100 for thresh in range(50, 100, 5)]

    def GetPascalVOCMetrics(self, boxes, thresh=0.5, method=EvaluationMethod.IoU):
        ret = []
        boxesByLabels = dictGrouping(boxes, key=lambda box: box.getClassId())
        labels = sorted(boxesByLabels.keys())

        for label in labels:
            boxesByDetectionMode = dictGrouping(
                boxesByLabels[label],
                key=lambda box: box.getBBType()
            )
            groundTruths = dictGrouping(
                boxesByDetectionMode[BBType.GroundTruth],
                key=lambda box: box.getImageName()
            )
            detections = sorted(
                boxesByDetectionMode[BBType.Detected],
                key=lambda box: box.getConfidence(),
                reverse=True
            )

            TP = np.repeat(False, len(detections))
            npos = len(boxesByDetectionMode[BBType.GroundTruth])
            visited = {k: np.repeat(False, len(v))
                for k, v in groundTruths.items()
            }

            for (i, detection) in enumerate(detections):
                imageName = detection.getImageName()
                associatedGts = groundTruths[imageName]

                if method == EvaluationMethod.IoU:
                    maxIoU = 0

                    for j, gt in enumerate(associatedGts):
                        iou = detection.iou(gt)

                        if iou > maxIoU:
                            maxIoU = iou
                            jmax = j

                    if maxIoU > thresh and not visited[imageName][jmax]:
                        visited[imageName][jmax] = True
                        TP[i] = True

                if method == EvaluationMethod.Distance:
                    minDist = distMin = sys.float_info.max
                    minImgSize = min(detection.getImageSize())
                    normThresh = thresh * minImgSize

                    for (j, gt) in enumerate(associatedGts):
                        dist = detection.distance(gt)

                        if dist < minDist:
                            minDist = dist
                            jmin = j

                    if minDist < normThresh and not visited[imageName][jmin]:
                        visited[imageName][jmin] = True
                        TP[i] = True

            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum([not tp for tp in TP])
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)

            ret.append({
                'class': label,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total detections': len(TP),
                'total TP': sum(TP),
                'total FP': sum([not tp for tp in TP])
            })

        return ret

    def getCocoMetrics(self, boundingBoxes):
        return [self.GetPascalVOCMetrics(boundingBoxes, thresh)
            for thresh in self.cocoThresholds]

    def getAP(self, boundingBoxes, thresh=0.5, method=EvaluationMethod.IoU):
        AP = [res["AP"]
            for res in self.GetPascalVOCMetrics(boundingBoxes, thresh, method)]
        return sum(AP) / len(AP) if AP else 0.0

    def getCocoAP(self, boundingBoxes):
        AP = [self.getAP(boundingBoxes, thresh)
            for thresh in self.cocoThresholds]
        return sum(AP) / len(AP) if AP else 0.0

    def printAPs(self, boxes):
        APs = [self.getAP(boxes, thresh)
            for thresh in self.cocoThresholds]
        cocoAP = sum(APs) / len(APs) if APs else 0.0

        print("mAP@.50: {:.2%}".format(APs[0]))
        print("mAP@.75: {:.2%}".format(APs[5]))
        print("coco AP: {:.2%}".format(cocoAP))

    def printAPsByClass(self, boxes, thresh=0.5, method=EvaluationMethod.IoU):
        if thresh is not None:
            metrics = self.GetPascalVOCMetrics(boxes, thresh, method)
            print("AP@{} by class:".format(thresh))
            for metric in metrics:
                label = metric["class"]
                AP = metric["AP"]
                totalPositive = metric["total positives"]
                totalDetections = metric["total detections"]
                TP = metric["total TP"]
                FP = metric["total FP"]
                print("  {:<13} - AP: {:.2%}  npos: {}  nDet: {}  TP: {}  FP: {}".format(label, AP, totalPositive, totalDetections, TP, FP))

    def PlotPrecisionRecallCurve(self,
                                 boundingBoxes,
                                 IOUThreshold=0.5,
                                 showAP=False,
                                 showInterpolatedPrecision=False,
                                 savePath=None,
                                 showGraphic=True):
        """PlotPrecisionRecallCurve
        Plot the Precision x Recall curve for a given class.
        Args:
            boundingBoxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOUThreshold (optional): IOU threshold indicating which detections will be considered
            TP or FP (default value = 0.5);
            method (default = EveryPointInterpolation): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
            interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EveryPointInterpolation"  (ElevenPointInterpolation).
            showAP (optional): if True, the average precision value will be shown in the title of
            the graph (default = False);
            showInterpolatedPrecision (optional): if True, it will show in the plot the interpolated
             precision (default = False);
            savePath (optional): if informed, the plot will be saved as an image in this path
            (ex: /home/mywork/ap.png) (default = None);
            showGraphic (optional): if True, the plot will be shown (default = True)
        Returns:
            A list of dictionaries. Each dictionary contains information and metrics of each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        results = self.GetPascalVOCMetrics(boundingBoxes, IOUThreshold)
        result = None
        # Each resut represents a class
        for result in results:
            if result is None:
                raise IOError('Error: Class %d could not be found.' % classId)

            classId = result['class']
            precision = result['precision']
            recall = result['recall']
            average_precision = result['AP']
            mpre = result['interpolated precision']
            mrec = result['interpolated recall']
            npos = result['total positives']
            total_tp = result['total TP']
            total_fp = result['total FP']

            plt.close()
            if showInterpolatedPrecision:
                if method == MethodAveragePrecision.EveryPointInterpolation:
                    plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
                elif method == MethodAveragePrecision.ElevenPointInterpolation:
                    # Uncomment the line below if you want to plot the area
                    # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                    # Remove duplicates, getting only the highest precision of each recall value
                    nrec = []
                    nprec = []
                    for idx in range(len(mrec)):
                        r = mrec[idx]
                        if r not in nrec:
                            idxEq = np.argwhere(mrec == r)
                            nrec.append(r)
                            nprec.append(max([mpre[int(id)] for id in idxEq]))
                    plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
            plt.plot(recall, precision, label='Precision')
            plt.xlabel('recall')
            plt.ylabel('precision')
            if showAP:
                ap_str = "{0:.2f}%".format(average_precision * 100)
                # ap_str = "{0:.4f}%".format(average_precision * 100)
                plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
            else:
                plt.title('Precision x Recall curve \nClass: %d' % classId)
            plt.legend(shadow=True)
            plt.grid()

            if savePath is not None:
                plt.savefig(os.path.join(savePath, classId + '.png'))
            if showGraphic is True:
                plt.show()
                # plt.waitforbuttonpress()
                plt.pause(0.05)
        return results

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]
