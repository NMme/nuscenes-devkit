# nuScenes dev-kit.
# Code written by Holger Caesar, Varun Bankiti, and Alex Lang, 2019.

import json
from typing import Any
import os

import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.constants import TP_METRICS, DETECTION_NAMES, DETECTION_COLORS, TP_METRICS_UNITS, \
    PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
#from nuscenes.eval.detection.data_classes import DetectionMetrics, DetectionMetricDataList
from data_classes import DetectionMetrics, DetectionMetricData, DetectionMetricDataList, DetectionBox
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, view_points, box_in_image
from PIL import Image, ImageOps

Axis = Any


def visualize_fp_detection(nusc: NuScenes,
                           sample_token: str,
                           box: Box,
                           savepath: str = None):
    # get sample and camera data tokens
    sample = nusc.get('sample', sample_token)
    cams = [sample['data'][key] for key in sample['data'].keys() if 'CAM' in key]

    # iterate over different visibility levels
    vis_levels = [BoxVisibility.ALL, BoxVisibility.ANY, BoxVisibility.NONE]
    matched_cam = dict()
    for vis_lvl in vis_levels:
        # check all cams
        for cam_token in cams:
            eval_box = box.copy()
            # gather sensor data
            sample_data = nusc.get('sample_data', cam_token)
            calibrated_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])

            # move box to ego vehicle coordinate system
            eval_box.translate(-np.array(pose_record['translation']))
            eval_box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            eval_box.translate(-np.array(calibrated_data['translation']))
            eval_box.rotate(Quaternion(calibrated_data['rotation']).inverse)

            intrinsic = np.array(calibrated_data['camera_intrinsic'])
            imsize = tuple((sample_data['width'], sample_data['height']))
            # check if box is in image
            if box_in_image(eval_box, intrinsic, imsize=imsize, vis_level=vis_lvl):
                matched_cam = {'cam_token': cam_token, 'sample_data': sample_data, 'calibrated_data': calibrated_data,
                               'intrinsic': intrinsic}
                box = eval_box.copy()
                break
        if matched_cam:
            break

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    img = Image.open(os.path.join(nusc.dataroot, matched_cam['sample_data']['filename']))
    ax.imshow(img, origin='upper')
    box.render(ax, view=matched_cam['intrinsic'], normalize=True, colors=('r', 'r', 'r'))
    #ax.set_xlim(0, matched_cam['sample_data']['width'])
    #ax.set_ylim(0, matched_cam['sample_data']['height'])

    plt.axis('off')
    plt.title('FP Detection w/ confidence: %4f' % box.score)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def class_pr_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_precision: float,
                   min_recall: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot a precision recall curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: The detection class.
    :param min_precision:
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Precision', xlim=1,
                        ylim=1, min_precision=min_precision, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_fdrr_curve(md_list: DetectionMetricDataList,
                     metrics: DetectionMetrics,
                     detection_name: str,
                     min_fdr: float,
                     min_recall: float,
                     savepath: str = None,
                     ax: Axis = None) -> None:
    """
    Plot a false-dicovery-rate recall curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: The detection class.
    :param min_precision:
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='FDR', xlim=1,
                        ylim=1, min_precision=min_fdr, min_recall=min_recall)

    # Get recall vs precision values of given class for each distance threshold.
    data = md_list.get_class_data(detection_name)

    # Plot the recall vs. precision curve for each distance threshold.
    for md, dist_th in data:
        md: DetectionMetricData
        ap = metrics.get_label_ap(detection_name, dist_th)
        fdr = 1.0 - md.precision
        ax.plot(md.recall, fdr, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.
    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_fdr_dist_curve(md_list: DetectionMetricDataList,
                         detection_name: str,
                         dist_th: list,
                         x_lim: int = 1,
                         y_lim: int = 1,
                         savepath: str = None) -> None:
    """
    Plot the FDR for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: name of the detection
    :param dist_th: Distance threshold for matching.
    :param x_lim: Upper limit for x-axis
    :param y_lim: Upper limit for y-axis
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    ax = setup_axis(xlabel='distance threshold', ylabel='FDR', xlim=x_lim, ylim=y_lim, ax=ax)

    # Plot the distance vs. fdr curve for each detection class.
    fdr = []
    for dist in dist_th:
        data = md_list.get_dist_data(dist)
        fdr.append(*[1.0-mds.true_precision for mds, dn in data if dn == detection_name])
    ax.plot(dist_th, fdr, label='{}%'.format(PRETTY_DETECTION_NAMES[detection_name]),
            color=DETECTION_COLORS[detection_name])
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_fp_conf_curves(md_list: DetectionMetricDataList,
                         metrics: DetectionMetrics,
                         detection_name: str,
                         savepath: str = None) -> None:
    """
    Plot the number of false positives for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param detection_name: name of the detection
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    fig, ax = plt.subplots()
    ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='confidence scores', ylabel='#FP')

    data = md_list.get_class_data(detection_name)
    for md, dist_th in data:
        max_bin = int(np.max(md.true_confidence)*10)
        bins = np.arange(max_bin + 2) * 0.1
        bin_ind = np.searchsorted(md.true_confidence[::-1], bins, side='left')
        for i, idx in enumerate(bin_ind):
            if idx == len(md.true_confidence):
                bin_ind[i] = idx - 1
        bin_fp = np.diff(md.fp[bin_ind])
        bin_tp = np.diff(md.tp[bin_ind])
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(bin_tp/bin_fp, label='Dist. : {}, AP: {:.1f}'.format(dist_th, ap * 100))

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_fdr_conf_hist(md_list: DetectionMetricDataList,
                        detection_name: str,
                        dist_th: float,
                        savepath: str = None) -> None:
    """
    Plot the FDR for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: name of the detection
    :param dist_th: Distance threshold for matching.
    :param x_lim: Upper limit for x-axis
    :param y_lim: Upper limit for y-axis
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # get the specified data metric
    md = md_list.get_class_dist_data(detection_name, dist_th)

    # setup and fill bins for the histogram
    bins = np.arange(11) * 0.1
    fig, ax = plt.subplots()
    ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='confidence scores', ylabel='#FP')
    x = [md.true_confidence[1:][np.diff(md.fp).astype(np.bool)], md.true_confidence[1:][np.diff(md.tp).astype(np.bool)]]
    n, bins, patches = ax.hist(x, bins, histtype='barstacked', stacked=True)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_fdr_conf_hist2(md_list: DetectionMetricDataList,
                         metrics: DetectionMetrics,
                         detection_name: str,
                         dist_th: list,
                         bin_size: float = 0.1,
                         savepath: str = None) -> None:
    """
    Plot the FDR for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param detection_name: name of the detection
    :param dist_th: Distance threshold for matching.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # setup plot axis
    fig, ax = plt.subplots()
    ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='confidence scores', ylabel='#FP')

    # setup and fill plot data
    bins = int(1/bin_size)
    data = np.zeros((bins, len(dist_th) + 1))
    ap = np.zeros(len(dist_th))
    for idx, (md, dist) in enumerate(md_list.get_class_data(detection_name)):
        hist_fp, be = np.histogram(md.true_confidence[:-1][np.diff(md.fp).astype(np.bool)], bins, range=(0.0, 1.0))
        # set upper limit for first iteration
        if idx == 0:
            hist_tp, be = np.histogram(md.true_confidence[:-1][np.diff(md.tp).astype(np.bool)], bins, range=(0.0, 1.0))
            data[:, idx] = hist_fp + hist_tp
        # save number of false positives
        data[:, idx+1] = hist_fp
        ap[idx] = metrics.get_label_ap(detection_name, dist)

    # plot data as bar graph
    bin_mids = np.linspace(bin_size/2, 1-(bin_size/2), data.shape[0])
    y_offset = np.zeros(len(bin_mids))
    #colors = plt.cm.winter(np.linspace(0, 1, bins+1))
    colors = plt.cm.get_cmap('tab20c', bins+1).colors
    data = np.fliplr(data)
    len_d = len(data.T)
    for i, row in enumerate(data.T):
        if i == len_d-1:
            ax.bar(bin_mids, row-y_offset, width=bin_size, bottom=y_offset, align='center', color=colors[i],
                   label='# total Detections')
        else:
            ax.bar(bin_mids, row-y_offset, width=bin_size, bottom=y_offset, align='center', color=colors[i],
                   label='#FP @ Dist. : {}, AP: {:.1f}'.format(dist_th[len_d-2-i], ap[len_d-2-i] * 100))
        #y_offset = y_offset + row
        y_offset = row

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def class_fdr_conf_hist3(md_list: DetectionMetricDataList,
                         metrics: DetectionMetrics,
                         detection_name: str,
                         dist_th: list,
                         bin_size: float = 0.1,
                         savepath: str = None) -> None:
    """
    Plot the FDR for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param detection_name: name of the detection
    :param dist_th: Distance threshold for matching.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # setup plot axis
    fig, ax = plt.subplots()
    ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='confidence scores', ylabel='rel. #FP')

    # setup and fill plot data
    bins = int(1/bin_size)
    data = np.zeros((bins, len(dist_th) + 1))
    ap = np.zeros(len(dist_th))
    for idx, (md, dist) in enumerate(md_list.get_class_data(detection_name)):
        hist_fp, be = np.histogram(md.true_confidence[:-1][np.diff(md.fp).astype(np.bool)], bins, range=(0.0, 1.0))
        # set upper limit for first iteration
        if idx == 0:
            hist_tp, be = np.histogram(md.true_confidence[:-1][np.diff(md.tp).astype(np.bool)], bins, range=(0.0, 1.0))
            data[:, idx] = hist_fp + hist_tp
        # save number of false positives
        data[:, idx+1] = hist_fp
        ap[idx] = metrics.get_label_ap(detection_name, dist)

    # calculate relative values for data
    data_rel = np.zeros(data.T.shape)
    np.divide(data.T, data[:, 0], out=data_rel, where=data.T != 0.0)
    data_rel = np.flipud(data_rel)

    # plot data as bar graph
    bin_mids = np.linspace(bin_size/2, 1-(bin_size/2), data.shape[0])
    y_offset = np.zeros(len(bin_mids))
    #colors = plt.cm.winter(np.linspace(0, 1, bins+1))
    colors = plt.cm.get_cmap('tab20c', bins+1).colors
    len_d = len(data_rel)
    for i, row in enumerate(data_rel):
        if i == len_d-1:
            ax.bar(bin_mids, row-y_offset, width=bin_size, bottom=y_offset, align='center', color=colors[i],
                   label='all Detections (FP + TP)')
        else:
            ax.bar(bin_mids, row-y_offset, width=bin_size, bottom=y_offset, align='center', color=colors[i],
                   label='rel. #FP @ Dist. : {}, AP: {:.1f}'.format(dist_th[len_d-2-i], ap[len_d-2-i] * 100))
        #y_offset = y_offset + row
        y_offset = row

    ax.legend(loc='best')
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def fdr_dist_curves(md_list: DetectionMetricDataList,
                    dist_th: list,
                    x_lim: int = 1,
                    y_lim: int = 1,
                    savepath: str = None) -> None:
    """
    Plot the FDR for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name: name of the detection
    :param dist_th: Distance threshold for matching.
    :param x_lim: Upper limit for x-axis
    :param y_lim: Upper limit for y-axis
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    ax = setup_axis(xlabel='distance threshold', ylabel='FDR', xlim=x_lim, ylim=y_lim, ax=ax)

    # Plot the distance vs. fdr curve for each detection class.
    categories = md_list.get_categories()
    for detection_name in categories:
        fdr = []
        for dist in dist_th:
            data = md_list.get_dist_data(dist)
            fdr.append(*[1.0-mds.true_precision for mds, dn in data if dn == detection_name])
        ax.plot(dist_th, fdr, label='{}'.format(PRETTY_DETECTION_NAMES[detection_name]),
                color=DETECTION_COLORS[detection_name])
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def dist_pr_curve(md_list: DetectionMetricDataList,
                  metrics: DetectionMetrics,
                  dist_th: float,
                  min_precision: float,
                  min_recall: float,
                  savepath: str = None) -> None:
    """
    Plot the PR curves for different distance thresholds.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param dist_th: Distance threshold for matching.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Prepare axis.
    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]},
                                  figsize=(7.5, 5))
    ax = setup_axis(xlabel='Recall', ylabel='Precision',
                    xlim=1, ylim=1, min_precision=min_precision, min_recall=min_recall, ax=ax)

    # Plot the recall vs. precision curve for each detection class.
    data = md_list.get_dist_data(dist_th)
    for md, detection_name in data:
        md = md_list[(detection_name, dist_th)]
        ap = metrics.get_label_ap(detection_name, dist_th)
        ax.plot(md.recall, md.precision, label='{}: {:.1f}%'.format(PRETTY_DETECTION_NAMES[detection_name], ap * 100),
                color=DETECTION_COLORS[detection_name])
    hx, lx = ax.get_legend_handles_labels()
    lax.legend(hx, lx, borderaxespad=0)
    lax.axis("off")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()


def summary_plot(md_list: DetectionMetricDataList,
                 metrics: DetectionMetrics,
                 min_precision: float,
                 min_recall: float,
                 dist_th_tp: float,
                 savepath: str = None) -> None:
    """
    Creates a summary plot with PR and TP curves for each class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param min_precision: Minimum precision value.
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    n_classes = len(DETECTION_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes))
    for ind, detection_name in enumerate(DETECTION_NAMES):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])
        ax1.set_ylabel('{} \n \n Precision'.format(PRETTY_DETECTION_NAMES[detection_name]), size=20)

        ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name,  min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


def detailed_results_table_tex(metrics_path: str, output_path: str) -> None:
    """
    Renders a detailed results table in tex.
    :param metrics_path: path to a serialized DetectionMetrics file.
    :param output_path: path to the output file.
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    tex = ''
    tex += '\\begin{table}[]\n'
    tex += '\\small\n'
    tex += '\\begin{tabular}{| c | c | c | c | c | c | c |} \\hline\n'
    tex += '\\textbf{Class}    &   \\textbf{AP}  &   \\textbf{ATE} &   \\textbf{ASE} & \\textbf{AOE}   & ' \
           '\\textbf{AVE}   & ' \
           '\\textbf{AAE}   \\\\ \\hline ' \
           '\\hline\n'
    for name in DETECTION_NAMES:
        ap = np.mean(metrics['label_aps'][name].values()) * 100
        ate = metrics['label_tp_errors'][name]['trans_err']
        ase = metrics['label_tp_errors'][name]['scale_err']
        aoe = metrics['label_tp_errors'][name]['orient_err']
        ave = metrics['label_tp_errors'][name]['vel_err']
        aae = metrics['label_tp_errors'][name]['attr_err']
        tex_name = PRETTY_DETECTION_NAMES[name]
        if name == 'traffic_cone':
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   N/A  &   N/A  &   N/A  \\\\ \\hline\n'.format(
                tex_name, ap, ate, ase)
        elif name == 'barrier':
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   N/A  &   N/A  \\\\ \\hline\n'.format(
                tex_name, ap, ate, ase, aoe)
        else:
            tex += '{}  &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  \\\\ ' \
                   '\\hline\n'.format(tex_name, ap, ate, ase, aoe, ave, aae)

    map_ = metrics['mean_ap']
    mate = metrics['tp_errors']['trans_err']
    mase = metrics['tp_errors']['scale_err']
    maoe = metrics['tp_errors']['orient_err']
    mave = metrics['tp_errors']['vel_err']
    maae = metrics['tp_errors']['attr_err']
    tex += '\\hline {} &   {:.1f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  &   {:.2f}  \\\\ ' \
           '\\hline\n'.format('\\textbf{Mean}', map_, mate, mase, maoe, mave, maae)

    tex += '\\end{tabular}\n'

    # All one line
    tex += '\\caption{Detailed detection performance on the val set. \n'
    tex += 'AP: average precision averaged over distance thresholds (%), \n'
    tex += 'ATE: average translation error (${}$), \n'.format(TP_METRICS_UNITS['trans_err'])
    tex += 'ASE: average scale error (${}$), \n'.format(TP_METRICS_UNITS['scale_err'])
    tex += 'AOE: average orientation error (${}$), \n'.format(TP_METRICS_UNITS['orient_err'])
    tex += 'AVE: average velocity error (${}$), \n'.format(TP_METRICS_UNITS['vel_err'])
    tex += 'AAE: average attribute error (${}$). \n'.format(TP_METRICS_UNITS['attr_err'])
    tex += 'nuScenes Detection Score (NDS) = {:.1f} \n'.format(metrics['nd_score'] * 100)
    tex += '}\n'

    tex += '\\end{table}\n'

    with open(output_path, 'w') as f:
        f.write(tex)
