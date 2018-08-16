import cv2
import numpy as np


def draw_pts(image, gt_pts=None, pred_pts=None, get_l1e=False):
    l1_distances = []
    pred_pts = np.transpose(pred_pts, [1, 0])
    shape = image.shape
    image = cv2.resize(image, (512,512), 0)
    pred_pts[:,0] *= 512/shape[1]
    pred_pts[:,1] *= 512/shape[0]

    if gt_pts is not None:

        gt_pts[:, 0] *= 512 / shape[1]
        gt_pts[:, 1] *= 512 / shape[0]

        for i, pt in enumerate(pred_pts):
            gpt = gt_pts[i]
            single_pt_distance = np.sqrt(np.square(pt[0] - gpt[0]) + np.square(pt[1] - gpt[1]))
            l1_distances.append(single_pt_distance)
            pred_pt = (int(pt[0]), int(pt[1]))
            grnd_pt = (int(gpt[0]), int(gpt[1]))
            cv2.circle(image, pred_pt, 2, (255, 0, 0))
            cv2.line(image, pred_pt, grnd_pt, (0, 0, 255))

        if get_l1e:
            return image, l1_distances
    else:
        for pt in pred_pts:
            pred_pt = (int(pt[0]), int(pt[1]))
            cv2.circle(image, pred_pt, 2, (255, 0, 0))
        return image