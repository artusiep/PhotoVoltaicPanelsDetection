from __future__ import annotations

import json
import os
import pickle
from typing import Union

import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union


class Report:
    PREDICTED_ID = 'pred_id'
    GROUND_TRUTH_ID = 'ground_truth_id'
    IOU = 'IOU'
    PREDICTED_RECT = 'predicted_rect'
    GROUND_TRUTH_RECT = 'ground_truth_rect'

    def __init__(self):
        self.ground_truth_rectangles: Union[None, list] = None
        self.prediction_rectangles: Union[None, list] = None
        self.df: Union[None, pd.DataFrame] = None
        self.report: Union[None, dict] = None

    def serialize_evaluation_result(self, path: str):
        if self.df is not None:
            self.df.to_pickle(f'{path}.ptl')
        else:
            print("Cannot serialize evaluation results as DataFrame is not known.")
            exit(1)

    @classmethod
    def deserialize_evaluation_result(cls, path: str) -> Report:
        report = cls()
        report.df = pd.read_pickle(f'{path}.ptl')
        report.prediction_rectangles = list(report.df[cls.PREDICTED_RECT])
        report.ground_truth_rectangles = list(report.df[cls.GROUND_TRUTH_RECT])
        return report

    def create_report(self) -> dict:
        sorted_df = self.df[[self.PREDICTED_ID, self.GROUND_TRUTH_ID, self.IOU]].sort_values(self.IOU, ascending=False)
        pred_id_dedup_df = sorted_df.drop_duplicates(subset=[self.PREDICTED_ID], keep='first')
        ground_truth_id_dedup_df = sorted_df.drop_duplicates(subset=[self.GROUND_TRUTH_ID], keep='first')

        found_1 = pred_id_dedup_df[pred_id_dedup_df[self.IOU] >= 0.6]
        found_2 = ground_truth_id_dedup_df[ground_truth_id_dedup_df[self.IOU] >= 0.6]
        if found_1.size != found_2.size:
            raise Exception("Results do not have sense. Number of founded rectangles should be equal/")
        badly_found = pred_id_dedup_df[pred_id_dedup_df[self.IOU] < 0.6]
        not_found = ground_truth_id_dedup_df[ground_truth_id_dedup_df[self.IOU] < 0.6]

        self.report = {
            'true_positive_df': found_1,
            'false_negative_df': not_found,
            'false_positive_df': badly_found,
            'ground_truth_rectangles_no': sorted_df[self.GROUND_TRUTH_ID].max() + 1,
            'pred_rectangles_no': sorted_df[self.PREDICTED_ID].max() + 1
        }
        return self.report

    def evaluate(self):
        raise NotImplementedError


class FromFileReport(Report):
    def __init__(self, label_file_path: str, prediction_file_path: str):
        super().__init__()
        self.prediction_file_path = prediction_file_path
        self.label_file_path = label_file_path

    def _get_ground_truth_rectangles(self) -> list:
        with open(self.label_file_path) as file:
            data = json.load(file)
            rectangles = [shape['points'] for shape in data['shapes']]
            self.round_truth_rectangles = rectangles
            return rectangles

    def _get_prediction_rectangles(self) -> list:
        with open(self.prediction_file_path, 'rb') as pickle_file:
            rectangles = pickle.load(pickle_file)['rectangles']
            self.prediction_rectangles = rectangles
            return rectangles

    def evaluate(self) -> pd.DataFrame:
        result_collector = []

        for pred_id, predicted_rect in enumerate(self._get_prediction_rectangles()):
            scaled_prediction_rect = predicted_rect / 3
            predicted_polygon = Polygon(scaled_prediction_rect)
            for ground_truth_id, ground_truth_rectangle in enumerate(self._get_ground_truth_rectangles()):
                label_polygon = Polygon(ground_truth_rectangle)
                polygons = [predicted_polygon, label_polygon]
                union = unary_union(polygons)
                intersection = label_polygon.intersection(predicted_polygon)
                IOU = intersection.area / union.area
                values = [pred_id, ground_truth_id, IOU, predicted_rect, ground_truth_rectangle]
                result_collector.append(values)

        self.df = pd.DataFrame(result_collector,
                               columns=[self.PREDICTED_ID, self.GROUND_TRUTH_ID, self.IOU, self.PREDICTED_RECT,
                                        self.GROUND_TRUTH_RECT])
        return self.df


path = "here"
if not os.path.isfile(f'{path}.ptl'):
    print("Creating report")
    report = FromFileReport("data/thermal/plasma-DJI_1_R (437).json", "data/result/plasma-DJI_1_R (437).pickle")
    report.evaluate()
    report.serialize_evaluation_result("here")

report2 = Report.deserialize_evaluation_result('here')
report3 = report2.create_report()
print(report3)
