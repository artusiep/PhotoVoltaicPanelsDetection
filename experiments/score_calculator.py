class ScoreCalculator:

    class Score:

        def __init__(self, true_positive_no: int, false_negative_no: int, false_positive_no: int):
            self.true_positive_no = true_positive_no
            self.false_negative_no = false_negative_no
            self.false_positive_no = false_positive_no
            self.precision = self.calculate_precision(true_positive_no, false_positive_no)
            self.recall = self.calculate_recall(true_positive_no, false_negative_no)
            self.f1_score = self.calculate_f1_score(self.precision, self.recall)

        @staticmethod
        def calculate_precision(true_positive_no: int, false_positive_no: int):
            return round(true_positive_no / (true_positive_no + false_positive_no), 3)

        @staticmethod
        def calculate_recall(true_positive_no: int, false_negative_no: int):
            return round(true_positive_no / (true_positive_no + false_negative_no), 3)

        @staticmethod
        def calculate_f1_score(precision, recall):
            return 2 * (precision * recall) / (precision + recall)

    def __init__(self, reports: list[dict]):
        self.reports = reports
        self.scores = []
        self.aggregated_score = None

    def calculate_scores(self):
        total_true_positives = 0
        total_false_negatives = 0
        total_false_positives = 0

        [print(report) for report in self.reports]

        for report in self.reports:
            ground_truth_rectangles_no = report['ground_truth_rectangles_no']
            if ground_truth_rectangles_no == 0:
                continue

            true_positive_no = report['true_positive_df'].size
            false_negative_no = report['false_negative_df'].size
            false_positive_no = report['false_positive_df'].size

            score = self.Score(true_positive_no, false_negative_no, false_positive_no)
            self.scores.append(score)

            total_true_positives += true_positive_no
            total_false_negatives += false_negative_no
            total_false_positives += false_positive_no

        self.aggregated_score = self.Score(total_true_positives, total_false_negatives, total_false_positives)
