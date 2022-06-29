from abc import ABC, abstractmethod
from optparse import Option
from typing import Union

import numpy as np
from numpy import ndarray


class Base(ABC):

    def __init__(self) -> None:
        super().__init__()


class ESet(Base):
    def __init__(self, data: Union[list, ndarray],
                 test_data: Union[list, ndarray] = None,
                 label: str = None, dim: int = 3, c: int = 0) -> None:
        super().__init__()
        self.data = data
        self.test_data = test_data
        self._center = np.zeros(dim)
        self.label = label
        self.c = c # stand for class or color

    def get_center(self):
        return self._center

    def center(self, v):
        self._center = v
        return self

    def to_json(self) -> dict:
        return {'data': self.data.tolist()}


class Link:

    def __init__(self, v1: ESet, v2: ESet) -> None:
        super().__init__()
        self.v1 = v1
        self.v2 = v2
        self.meta = {}

    def get_ax(self, idx: int):
        # if self.meta['points']:
        #     points: list = self.meta['points']
        #     return [points[0][idx], points[1][idx]]
        # else:
        return [self.v1.get_center()[idx], self.v2.get_center()[idx]]

    def default_title(self) -> str:
        return self.v1.label + '__' + self.v2.label

    def to_json(self) -> dict:
        return self.meta


class Generator(ABC):
    @abstractmethod
    def generate(self, samples_size: int) -> ndarray:
        pass


class NormalDistributor(Generator):

    def __init__(self, mean, cov, size: int) -> None:
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.size = size

    def generate(self, samples_size: int) -> ndarray:
        x: ndarray = np.random.multivariate_normal(self.mean, self.cov, size=samples_size) # size=self.size
        return x
        # return super().generate(samples_size)


class Space:
    def __init__(self) -> None:
        super().__init__()
        self.sets = []
        self.links = []


class BaseBinaryClassifier:

    def __init__(self) -> None:
        """
        clf stands for trained binary classifier wrapped into BaseBinaryClassifier to learn over classes represented by ESet
        acc stand for accuracy cached while building binary classifier
        """
        super().__init__()
        self.clf = None
        self.acc: float = 0

    def build_for(self, s1: ESet, s2: ESet) -> float:
        """
        :param s1: - first set (order is unimportant)
        :param s2: - second set (order is unimportant)
        :return: accuracy of binary classifier being built
        """
        return 0


class LinkClassifier:

    def __init__(self, classifier: BaseBinaryClassifier) -> None:
        super().__init__()
        self.classifier = classifier

    def apply(self, link: Link) -> float:
        return self.classifier.build_for(link.v1, link.v2)

