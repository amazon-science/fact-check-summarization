# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from . import BaseWrapperDataset


class TruncateLastElementDataset(BaseWrapperDataset):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        item_len = item.size(0)
        if item_len > 0:
            item = item[:-1]
        return item

    # @property
    # def sizes(self):
    #     return np.minimum(self.dataset.sizes, self.truncation_length)

    # def __len__(self):
    #     return len(self.dataset)
