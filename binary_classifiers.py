from typing import List

from dist_util import BaseBinaryClassifier, ESet

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np


class SVMBinaryClassifier(BaseBinaryClassifier):

    def _get_classification(self):
        return LinearSVC()

    def build_for(self, s1: ESet, s2: ESet) -> float:
        self.clf = make_pipeline(StandardScaler(), self._get_classification())
        X, y = self.prepare_dataset(s1.data, s2.data)
        r = self.clf.fit(X, y)

        X_t, y_t = self.prepare_dataset(s1.test_data, s2.test_data)
        y_res = self.clf.predict(X_t)
        self.acc: float = len([1 for i in range(len(y_res)) if y_res[i] == y_t[i]]) / len(y_res)
        return self.acc

    def prepare_dataset(self, n1: np.ndarray, n2: np.ndarray):
        return self.prep_dataset([n1, n2])

    def prep_dataset(self, items: List[np.ndarray]):
        y = []
        for i, item in enumerate(items):
            y += [i for _ in range(len(item))]
        X = np.concatenate(items)
        return X, y


class SVMRbfBinaryClassifier(SVMBinaryClassifier):

    def _get_classification(self):
        return SVC()


class RandomBinaryClassifier(BaseBinaryClassifier):
    def build_for(self, s1: ESet, s2: ESet) -> float:
        self.acc = float(np.random.rand())
        return self.acc
