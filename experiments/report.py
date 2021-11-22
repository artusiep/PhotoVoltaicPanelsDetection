from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Union, List

import cv2
import humanize as humanize
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from detector.utils.display import display_image_in_actual_size
from detector.utils.utils import iou_rectangle_annotated_photos, rectangle_annotated_photos

_t = humanize.i18n.activate("pl")


@dataclass()
class ExperimentsConfig:
    experiment_dir: str
    name: str
    iou_threshold: float = 0.60


@dataclass
class ReportResult:
    identifier: str
    true_positive_df: pd.DataFrame
    false_negative_df: pd.DataFrame
    false_positive_df: pd.DataFrame
    ground_truth_rectangles_no: int
    pred_rectangles_no: int
    ground_truth_rectangles: List[np.ndarray]
    pred_rectangles: List[np.ndarray]


class Report:
    PREDICTED_ID = 'pred_id'
    GROUND_TRUTH_ID = 'ground_truth_id'
    IOU = 'IOU'
    PREDICTED_RECT = 'predicted_rect'
    GROUND_TRUTH_RECT = 'ground_truth_rect'
    ALL_PREDICTED_RECT = 'all_predicted_rect'
    ALL_GROUND_TRUTH_RECT = 'all_ground_truth_rect'
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    DETECTION_DURATION = 'detection_duration'

    def __init__(self):
        self.identifier: Union[None, str] = None
        self.ground_truth_rectangles: Union[None, list] = None
        self.prediction_rectangles: Union[None, list] = None
        self.df: Union[None, pd.DataFrame] = None
        self.report: Union[None, dict] = None
        self.start_time: Union[None, datetime] = None
        self.end_time: Union[None, datetime] = None
        self.detection_duration: Union[None, timedelta] = None

    @staticmethod
    def __safe_create_path(path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def serialize_evaluation_result(self, path: str):
        self.__safe_create_path(path)
        if self.df is not None:
            with open(f'{path}.ptl', 'wb') as pickle_file:
                pickle.dump(self, pickle_file)
        else:
            print("Cannot serialize evaluation results as DataFrame is not known. Try to invoke 'evaluate' method")
            exit(1)

    @staticmethod
    def evaluation_result_exist(path):
        return os.path.isfile(f'{path}.ptl')

    def _set_identifier_based_on_path(self, path: str):
        self.identifier = Path(path).stem

    @classmethod
    def deserialize_evaluation_result(cls, path: str) -> Report:
        report = pd.read_pickle(f'{path}.ptl')
        return report

    def create_report(self, experiment_config: ExperimentsConfig) -> dict:
        sorted_df = self.df[[self.PREDICTED_ID, self.GROUND_TRUTH_ID, self.IOU, self.PREDICTED_RECT,
                             self.GROUND_TRUTH_RECT]].sort_values(self.IOU, ascending=False)
        pred_id_dedup_df = sorted_df.drop_duplicates(subset=[self.PREDICTED_ID], keep='first')
        ground_truth_id_dedup_df = sorted_df.drop_duplicates(subset=[self.GROUND_TRUTH_ID], keep='first')

        dedup_df = sorted_df.drop_duplicates(subset=[self.PREDICTED_ID], keep='first').drop_duplicates(
            subset=[self.GROUND_TRUTH_ID], keep='first')

        true_positive = dedup_df[dedup_df[self.IOU] >= experiment_config.iou_threshold]
        badly_found = pred_id_dedup_df[~pred_id_dedup_df.isin(true_positive)].dropna()
        not_found = ground_truth_id_dedup_df[~ground_truth_id_dedup_df.isin(true_positive)].dropna()

        self.report = {
            'identifier': self.identifier,
            'true_positive_df': true_positive,
            'false_negative_df': not_found,
            'false_positive_df': badly_found,
            'ground_truth_rectangles_no': sorted_df[self.GROUND_TRUTH_ID].max() + 1 if sorted_df[
                                                                                           self.GROUND_TRUTH_ID].max() is not np.nan else 0,
            'pred_rectangles_no': sorted_df[self.PREDICTED_ID].max() + 1 if sorted_df[
                                                                                self.PREDICTED_ID].max() is not np.nan else 0,
            'ground_truth_rectangles': self.ground_truth_rectangles,
            'pred_rectangles': self.prediction_rectangles,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'detection_duration': self.detection_duration
        }
        return self.report

    def evaluate(self):
        raise NotImplementedError


class FromFileReport(Report):
    def __init__(self, ground_truth_file_path: str, prediction_file_path: str):
        super().__init__()
        self._set_identifier_based_on_path(prediction_file_path)
        self.prediction_file_path = prediction_file_path
        self.label_file_path = ground_truth_file_path
        self._get_ground_truth_rectangles()
        self._get_prediction_rectangles()

    def _get_ground_truth_rectangles(self) -> list:
        with open(self.label_file_path) as file:
            data = json.load(file)
            rectangles = [shape['points'] for shape in data['shapes']]
            self.ground_truth_rectangles = rectangles
            return rectangles

    @staticmethod
    def _get_tag_content(pickle_content: dict, field: str):
        if field in pickle_content:
            return pickle_content[field]

    def _get_prediction_rectangles(self) -> list:
        with open(self.prediction_file_path, 'rb') as pickle_file:
            pickle_content = pickle.load(pickle_file)
            self.prediction_rectangles = pickle_content['rectangles']
            if 'tags' in pickle_content:
                tags = pickle_content['tags']
                self.detection_duration = self._get_tag_content(tags, 'detection_duration')
                self.start_time = self._get_tag_content(tags, 'start_time')
                self.end_time = self._get_tag_content(tags, 'end_time')
            return pickle_content['rectangles']

    def evaluate(self) -> pd.DataFrame:
        result_collector = []

        for pred_id, predicted_rect in enumerate(self.prediction_rectangles):
            scaled_prediction_rect = predicted_rect
            predicted_polygon = Polygon(scaled_prediction_rect)
            for ground_truth_id, ground_truth_rectangle in enumerate(self.ground_truth_rectangles):
                label_polygon = Polygon(ground_truth_rectangle)
                polygons = [predicted_polygon, label_polygon]
                union = unary_union(polygons)
                intersection = label_polygon.intersection(predicted_polygon)
                IOU = intersection.area / union.area
                values = [pred_id, ground_truth_id, IOU, predicted_rect.tolist(), ground_truth_rectangle,
                          self.prediction_rectangles, self.ground_truth_rectangles, self.detection_duration,
                          self.start_time, self.end_time]
                result_collector.append(values)

        self.df = pd.DataFrame(result_collector,
                               columns=[self.PREDICTED_ID, self.GROUND_TRUTH_ID, self.IOU, self.PREDICTED_RECT,
                                        self.GROUND_TRUTH_RECT, self.ALL_PREDICTED_RECT, self.ALL_GROUND_TRUTH_RECT,
                                        self.START_TIME, self.END_TIME, self.DETECTION_DURATION])
        return self.df


class ReportGenerator:
    def __init__(self, ground_truth_paths, predictions_paths):
        self.predictions_paths = sorted(predictions_paths)
        self.ground_truth_paths = sorted(ground_truth_paths)
        self.reports_objs = None
        self.generated_reports = None
        self.errors = 0

    def generate(self, experiment_config: ExperimentsConfig):
        report_pairs = []
        for prediction_path in self.predictions_paths:
            for ground_truth_path in self.ground_truth_paths:
                if Path(prediction_path).stem != Path(ground_truth_path).stem:
                    continue
                report_pairs.append((ground_truth_path, prediction_path))
        self.reports_objs = (FromFileReport(*report_pair) for report_pair in report_pairs)
        self.generated_reports = (self.generate_one_report(report_obj, experiment_config) for report_obj in
                                  self.reports_objs)
        return self.generated_reports

    def generate_one_report(self, report: FromFileReport, experiment_config: ExperimentsConfig) -> Union[dict, None]:
        evaluation_result_path = f"data/{experiment_config.experiment_dir}/checkpoint/{report.identifier}"
        if not report.evaluation_result_exist(evaluation_result_path):
            # print(f"Creating report for {report.identifier}")
            try:
                report.evaluate()
            except Exception as e:
                self.errors = self.errors + 1
                print(f"Report evaluation of image {report.identifier} failed due to: {e}")
            report.serialize_evaluation_result(evaluation_result_path)
        else:
            # print(f"Utilising saved report for {report.identifier}")
            report = Report.deserialize_evaluation_result(evaluation_result_path)
        return report.create_report(experiment_config)


def f1score(tp, fp, fn):
    return tp / (tp + (fp + fn) / 2)


def rectangles_f1score(tp_rects, fp_rects, fn_rects):
    if (len(tp_rects) + 1 / 2 * (len(fp_rects) + len(fn_rects))) != 0:
        return (len(tp_rects) / (
                len(tp_rects) + 1 / 2 * (len(fp_rects) + len(fn_rects))))
    else:
        return 1

def display_selected_image(generated_report, all_rects, base_image, image_id):
    if generated_report['identifier'] == image_id:
        display_image_in_actual_size(iou_rectangle_annotated_photos(all_rects, base_image), 'result.png')
        display_image_in_actual_size(
            rectangle_annotated_photos(list(np.array(generated_report['ground_truth_rectangles'])), base_image),
            'ground_truth_rectangles.png')
        display_image_in_actual_size(
            rectangle_annotated_photos(list(np.array(generated_report['pred_rectangles'])), base_image),
            'pred_rectangles.png')

def main(experiments_config):
    report_generator = ReportGenerator(glob.glob('data/thermal-modules-final/*.json'),
                                       glob.glob(f'data/{experiments_config.experiment_dir}/result/*.pickle'))
    generated_reports = report_generator.generate(experiments_config)
    fscores = []
    all_true_positive_rects = 0
    all_false_negative_rects = 0
    all_false_positive_rects = 0
    ground_truth_images = 0
    all_rectangles = []

    for generated_report in generated_reports:
        path = f"data/{experiments_config.experiment_dir}/thermal/{generated_report['identifier']}.JPG"
        base_image = cv2.imread(path)
        true_positive_rects = list(generated_report['true_positive_df']
                                   [['ground_truth_rect', 'predicted_rect', 'IOU']].itertuples(index=False, name=None))
        false_negative_rects = list(generated_report['false_negative_df']
                                    [['ground_truth_rect', 'predicted_rect', 'IOU']].itertuples(index=False, name=None))
        false_positive_rects = list(generated_report['false_positive_df']
                                    [['ground_truth_rect', 'predicted_rect', 'IOU']].itertuples(index=False, name=None))
        all_rects = true_positive_rects + false_positive_rects + false_negative_rects
        display_selected_image(generated_report, all_rects, base_image, 'plasma-DJI_2_R (904)')
        all_true_positive_rects = all_true_positive_rects + len(true_positive_rects)
        all_false_negative_rects = all_false_negative_rects + len(false_negative_rects)
        all_false_positive_rects = all_false_positive_rects + len(false_positive_rects)
        ground_truth_images = len(generated_report['ground_truth_rectangles']) + ground_truth_images
        fscores.append(rectangles_f1score(true_positive_rects, false_positive_rects, false_negative_rects))
        all_rectangles.append(generated_report)

    print(f"{experiments_config.name} END with {report_generator.errors} errors ")
    overall_f1score = f1score(all_true_positive_rects, all_false_positive_rects, all_false_negative_rects)
    print(f"Overall f1score: {overall_f1score}")
    median_f1score = median(fscores)
    print(f"Median of f1score: {median_f1score}")
    detection_start_time = min([report['start_time'] for report in all_rectangles])
    detection_end_time = max([report['end_time'] for report in all_rectangles])
    print(f"Detection start time: {detection_start_time}, detection end time: {detection_end_time}")
    detection_duration = detection_end_time - detection_start_time
    print(f"Overall detection duration {detection_duration}")
    median_duration = median([report['detection_duration'] for report in all_rectangles])
    print(f"Median duration time: {median_duration}")
    mean_duration = sum([report['detection_duration'] for report in all_rectangles], timedelta()) / len(all_rectangles)
    print(f"Mean duration time: {mean_duration}")
    print(
        f"Percentiles: 99 - {np.nanpercentile(fscores, 99)}, "
        f"95 - {np.nanpercentile(fscores, 95)}, "
        f"90 - {np.nanpercentile(fscores, 90)}, "
        f"75 - {np.nanpercentile(fscores, 75)}, "
        f"50 - {np.nanpercentile(fscores, 50)}, "
        f"25 - {np.nanpercentile(fscores, 25)}"
    )
    print(
        f"{median_f1score:.4f} & {overall_f1score:.4f} & {humanize.precisedelta(detection_duration)} & {humanize.precisedelta(median_duration)}".replace(
            '.', ','))
    return fscores


def print_histogram(bins, fscores, legend, filename, show=True):
    fig, ax = plt.subplots(figsize=(16 / 2, 20 / 2))
    a_heights, a_bins = np.histogram(fscores[0], bins=bins)
    b_heights, b_bins = np.histogram(fscores[1], bins=bins)
    c_heights, c_bins = np.histogram(fscores[2], bins=bins)
    width = (a_bins[1] - a_bins[0]) / 4
    ax.bar(a_bins[:-1] - width, a_heights, width=width, color='royalblue')
    ax.bar(b_bins[:-1], b_heights, width=width, color='orange')
    ax.bar(c_bins[:-1] + width, c_heights, width=width, color='forestgreen')
    ax.set_xticks(np.arange(0, 1.0, 0.1))
    ax.set_xticklabels([f"{(x / bins):.1f}-{(x / bins + 1 / bins):.1f}".replace('.', ',') for x in range(0, bins)],
                       rotation=45, fontsize='large')
    ax.set_yticks(np.arange(0, 275, 25))
    ax.set_yticklabels(np.arange(0, 275, 25), fontsize='large')
    plt.style.context('seaborn-deep')
    plt.title('')
    plt.xlabel('F1 Score', fontsize='x-large')
    plt.ylabel('Liczba przypadków', fontsize='x-large')
    print()
    plt.grid(axis='both', alpha=0.75)
    plt.legend(legend, fontsize='x-large')
    if show:
        plt.show()
    else:
        plt.savefig(f'{filename}.png')


def plot_single(df, color, filename, cumulative=False, show=True):
    ax = df.plot.hist(color=color, cumulative=cumulative)

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 275, 25))

    plt.style.context('seaborn-deep')
    plt.title('')
    plt.xlabel('F1 Score')
    plt.ylabel('Liczba przypadków')
    plt.grid(axis='both', alpha=0.75)
    plt.legend(loc='upper left')
    if show:
        plt.show()
    else:
        plt.savefig(f'{filename}.png')


if __name__ == '__main__':
    configs = [
        ExperimentsConfig(experiment_dir='single-non-ml', name='Single Non ML'),
        ExperimentsConfig(experiment_dir='single-4unet', name='Single 4 Unet'),
        ExperimentsConfig(experiment_dir='single-linknet', name='Single Linknet'),
        ExperimentsConfig(experiment_dir='mutli-non-ml', name='Multithreading Non ML'),
        ExperimentsConfig(experiment_dir='multiproc-4unet', name='Multithreading 4 Unet'),
        ExperimentsConfig(experiment_dir='multi-linknet', name='Multithreading Linknet'),
    ]
    fscores = [main(config) for config in configs]

    import matplotlib.pyplot as plt

    np.array(fscores)
    print_histogram(10, fscores, ['zwyczajna', 'autorska', 'linknet'], filename="data/60/overview", show=False)
    plot_single(pd.DataFrame({'zwyczajna': fscores[0]}), color='royalblue', filename="data/60/zwyczajna", show=False)
    plot_single(pd.DataFrame({'autorska': fscores[1]}), color='orange', filename="data/60/autorska", show=False)
    plot_single(pd.DataFrame({'linknet': fscores[2]}), color='forestgreen', filename="data/60/linknet", show=False)

    configs = [
        ExperimentsConfig(experiment_dir='single-non-ml', name='Single Non ML', iou_threshold=0.75),
        ExperimentsConfig(experiment_dir='single-4unet', name='Single 4 Unet', iou_threshold=0.75),
        ExperimentsConfig(experiment_dir='single-linknet', name='Single Linknet', iou_threshold=0.75),
        ExperimentsConfig(experiment_dir='mutli-non-ml', name='Multithreading Non ML'),
        ExperimentsConfig(experiment_dir='multiproc-4unet', name='Multithreading 4 Unet'),
        ExperimentsConfig(experiment_dir='multi-linknet', name='Multithreading Linknet'),
    ]
    fscores = [main(config) for config in configs]

    np.array(fscores)
    print_histogram(10, fscores, ['zwyczajna', 'autorska', 'linknet'], filename="data/75/overview", show=False)
    plot_single(pd.DataFrame({'zwyczajna': fscores[0]}), color='royalblue', filename="data/75/zwyczajna", show=False)
    plot_single(pd.DataFrame({'autorska': fscores[1]}), color='orange', filename="data/75/autorska", show=False)
    plot_single(pd.DataFrame({'linknet': fscores[2]}), color='forestgreen', filename="data/75/linknet", show=False)
