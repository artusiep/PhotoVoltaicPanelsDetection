from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

from detector.utils.display import display_image_in_actual_size
from detector.utils.utils import iou_rectangle_annotated_photos


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

    def __init__(self):
        self.identifier: Union[None, str] = None
        self.ground_truth_rectangles: Union[None, list] = None
        self.prediction_rectangles: Union[None, list] = None
        self.df: Union[None, pd.DataFrame] = None
        self.report: Union[None, dict] = None

    @staticmethod
    def __safe_create_path(path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def serialize_evaluation_result(self, path: str):
        self.__safe_create_path(path)
        if self.df is not None:
            self.df.to_pickle(f'{path}.ptl')
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
        report = cls()
        report._set_identifier_based_on_path(path)
        report.df = pd.read_pickle(f'{path}.ptl')

        def save_first_elem(df: pd.DataFrame):
            try:
                return df.iloc[0]
            except IndexError:
                return []

        report.prediction_rectangles = save_first_elem(report.df[cls.ALL_PREDICTED_RECT])
        report.ground_truth_rectangles = save_first_elem(report.df[cls.ALL_PREDICTED_RECT])
        return report

    def create_report(self) -> dict:
        sorted_df = self.df[[self.PREDICTED_ID, self.GROUND_TRUTH_ID, self.IOU, self.PREDICTED_RECT,
                             self.GROUND_TRUTH_RECT]].sort_values(self.IOU, ascending=False)
        pred_id_dedup_df = sorted_df.drop_duplicates(subset=[self.PREDICTED_ID], keep='first')
        ground_truth_id_dedup_df = sorted_df.drop_duplicates(subset=[self.GROUND_TRUTH_ID], keep='first')

        found_1 = pred_id_dedup_df[pred_id_dedup_df[self.IOU] >= 0.6]
        found_2 = ground_truth_id_dedup_df[ground_truth_id_dedup_df[self.IOU] >= 0.6]
        if found_1.size != found_2.size:
            raise Exception("Results do not have sense. Number of founded rectangles should be equal/")
        badly_found = pred_id_dedup_df[pred_id_dedup_df[self.IOU] < 0.6]
        not_found = ground_truth_id_dedup_df[ground_truth_id_dedup_df[self.IOU] < 0.6]

        self.report = {
            'identifier': self.identifier,
            'true_positive_df': found_1,
            'false_negative_df': not_found,
            'false_positive_df': badly_found,
            'ground_truth_rectangles_no': sorted_df[self.GROUND_TRUTH_ID].max() + 1 if sorted_df[
                                                                                           self.GROUND_TRUTH_ID].max() is not np.nan else 0,
            'pred_rectangles_no': sorted_df[self.PREDICTED_ID].max() + 1 if sorted_df[
                                                                                self.PREDICTED_ID].max() is not np.nan else 0,
            'ground_truth_rectangles': self.ground_truth_rectangles,
            'pred_rectangles': self.prediction_rectangles,
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

    def _get_prediction_rectangles(self) -> list:
        with open(self.prediction_file_path, 'rb') as pickle_file:
            rectangles = pickle.load(pickle_file)['rectangles']
            self.prediction_rectangles = rectangles
            return rectangles

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
                          self.prediction_rectangles, self.ground_truth_rectangles]
                result_collector.append(values)

        self.df = pd.DataFrame(result_collector,
                               columns=[self.PREDICTED_ID, self.GROUND_TRUTH_ID, self.IOU, self.PREDICTED_RECT,
                                        self.GROUND_TRUTH_RECT, self.ALL_PREDICTED_RECT, self.ALL_GROUND_TRUTH_RECT])
        return self.df


class ReportGenerator:
    def __init__(self, ground_truth_paths, predictions_paths):
        self.predictions_paths = sorted(predictions_paths)
        self.ground_truth_paths = sorted(ground_truth_paths)
        self.reports_objs = None
        self.generated_reports = None
        self.errors = 0

    def generate(self):
        report_pairs = []
        for prediction_path in self.predictions_paths:
            for ground_truth_path in self.ground_truth_paths:
                if Path(prediction_path).stem != Path(ground_truth_path).stem:
                    continue
                report_pairs.append((ground_truth_path, prediction_path))
        self.reports_objs = (FromFileReport(*report_pair) for report_pair in report_pairs)
        self.generated_reports = (self.generate_one_report(report_obj) for report_obj in self.reports_objs)
        return self.generated_reports

    def generate_one_report(self, report: FromFileReport):
        evaluation_result_path = f"data/checkpoint/{report.identifier}"
        if not report.evaluation_result_exist(evaluation_result_path):
            print(f"Creating report for {report.identifier}")
            try:
                report.evaluate()
            except Exception as e:
                self.errors = self.errors + 1
                print(f"Report evaluation of image {report.identifier} failed due to: {e}")
            report.serialize_evaluation_result(evaluation_result_path)
        else:
            print(f"Utilising saved report for {report.identifier}")
            report = Report.deserialize_evaluation_result(evaluation_result_path)
        return report.create_report()


if __name__ == '__main__':
    report_generator = ReportGenerator(glob.glob('data/thermal-modules/*.json'), glob.glob('data/result/*.pickle'))
    generated_reports = report_generator.generate()

    for generated_report in generated_reports:
        path = f"data/thermal/{generated_report['identifier']}.JPG"
        base_image = cv2.imread(path)
        true_positive_rects = list(generated_report['true_positive_df'][['ground_truth_rect', 'predicted_rect', 'IOU']].itertuples(index=False, name=None))
        false_negative_rects = list(generated_report['false_negative_df'][['ground_truth_rect', 'predicted_rect', 'IOU']].itertuples(index=False, name=None))
        false_positive_rects = list(generated_report['false_positive_df'][['ground_truth_rect', 'predicted_rect', 'IOU']].itertuples(index=False, name=None))
        all_rects = true_positive_rects + false_positive_rects + false_negative_rects
        display_image_in_actual_size(iou_rectangle_annotated_photos(all_rects, base_image))
    print(f"END with {report_generator.errors} errors ")
