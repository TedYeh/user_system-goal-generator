# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NLU Metric"""

import datasets
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from convlab.base_models.t5.nlu.serialization import deserialize_dialogue_acts


# TODO: Add BibTeX citation
_CITATION = """\
"""

_DESCRIPTION = """\
Metric to evaluate text-to-text models on the natural language understanding task.
"""

_KWARGS_DESCRIPTION = """
Calculates sequence exact match, dialog acts accuracy and f1
Args:
    predictions: list of predictions to score. Each predictions
        should be a string.
    references: list of reference for each prediction. Each
        reference should be a string.
Returns:
    seq_em: sequence exact match
    accuracy: dialog acts accuracy
    overall_f1: dialog acts overall f1
Examples:

    >>> nlu_metric = datasets.load_metric("nlu_metric.py")
    >>> predictions = ["[thank][general]{[][]}", "[inform][taxi]{[leave at][17:15]}"]
    >>> references = ["[thank][general]{[][]}", "[inform][train]{[leave at][17:15]}"]
    >>> results = nlu_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'seq_em': 0.5, 'accuracy': 0.5, 
    'overall_f1': 0.5, 'overall_precision': 0.5, 'overall_recall': 0.5}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NLUMetrics(datasets.Metric):
    """Metric to evaluate text-to-text models on the natural language understanding task."""

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            })
        )

    def count_confus(self, pred, golden, d_type):
        for ele in pred:
            if ele in golden:
                self.TP[d_type] += 1
            else:
                self.FP[d_type] += 1
        for ele in golden:
            if ele not in pred:
                self.FN[d_type] += 1

    def get_metrics(self, d_type):
        precision = 1.0 * self.TP[d_type] / (self.TP[d_type] + self.FP[d_type]) if self.TP[d_type] + self.FP[d_type] else 0.
        recall = 1.0 * self.TP[d_type] / (self.TP[d_type] + self.FN[d_type]) if self.TP[d_type] + self.FN[d_type] else 0.
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
        return f1, precision, recall
        
    def _compute(self, predictions, references):
        """Returns the scores: sequence exact match, dialog acts accuracy and f1"""
        seq_em = []
        acc = []
        self.TP = {'intent':0, 'domain':0, 'slot':0}
        self.FP = {'intent':0, 'domain':0, 'slot':0}
        self.FN = {'intent':0, 'domain':0, 'slot':0}
        f1_metrics = {'TP':0, 'FP':0, 'FN':0}

        for prediction, reference in zip(predictions, references):
            seq_em.append(prediction.strip()==reference.strip())
            pred_da = deserialize_dialogue_acts(prediction)
            gold_da = deserialize_dialogue_acts(reference)
            pred_intents, gold_intents = sorted(list({da['intent']for da in pred_da})), sorted(list({da['intent']for da in gold_da}))
            pred_slots, gold_slots = sorted(list({da['slot']for da in pred_da})), sorted(list({da['slot']for da in gold_da}))
            pred_domains, gold_domains = sorted(list({da['domain']for da in pred_da})), sorted(list({da['domain']for da in gold_da}))
            
            pred_da = sorted(list({(da['intent'], da['domain'], da['slot'], ''.join(da.get('value', '').split()).lower()) for da in pred_da}))
            gold_da = sorted(list({(da['intent'], da['domain'], da['slot'], ''.join(da.get('value', '').split()).lower()) for da in gold_da}))
            acc.append(pred_da==gold_da)
            self.count_confus(pred_domains, gold_domains, 'domain')
            self.count_confus(pred_intents, gold_intents, 'intent')
            self.count_confus(pred_slots, gold_slots, 'slot')
            for ele in pred_da:
                if ele in gold_da:
                    f1_metrics['TP'] += 1
                else:
                    f1_metrics['FP'] += 1
            for ele in gold_da:
                if ele not in pred_da:
                    f1_metrics['FN'] += 1

        overall_TP = f1_metrics.pop('TP')
        overall_FP = f1_metrics.pop('FP')
        overall_FN = f1_metrics.pop('FN')
        overall_precision = 1.0 * overall_TP / (overall_TP + overall_FP) if overall_TP + overall_FP else 0.
        overall_recall = 1.0 * overall_TP / (overall_TP + overall_FN) if overall_TP + overall_FN else 0.
        overall_f1 = 2.0 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall else 0.
        f1_metrics[f'overall_f1'] = overall_f1
        f1_metrics[f'overall_precision'] = overall_precision
        f1_metrics[f'overall_recall'] = overall_recall
        f1_metrics[f'intent_f1'], f1_metrics[f'intent_precision'], f1_metrics[f'intent_recall'] = \
        self.get_metrics('intent')
        f1_metrics[f'domain_f1'], f1_metrics[f'domain_precision'], f1_metrics[f'domain_recall'] = \
        self.get_metrics('domain')
        f1_metrics[f'slot_f1'], f1_metrics[f'slot_precision'], f1_metrics[f'slot_recall'] = \
        self.get_metrics('slot')
        return {
            "seq_em": sum(seq_em)/len(seq_em),
            "accuracy": sum(acc)/len(acc),
            **f1_metrics
        }
