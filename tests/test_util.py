# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import unittest

from sparkext.util import batched


class UtilTest(unittest.TestCase):
    def test_batched(self):
        pdf = pd.DataFrame([1,2,3,4,5,6,7,8,9,10])

        # no batching, expect 1 batch of 10
        batches = list(batched(pdf, -1))
        self.assertEqual(1, len(batches))
        self.assertEqual(10, len(batches[0]))

        # no batching, expect 1 batch of 10
        batches = list(batched(pdf, 0))
        self.assertEqual(1, len(batches))
        self.assertEqual(10, len(batches[0]))

        # batch_size = 1, expect 10 batches of 1
        batches = list(batched(pdf, 1))
        self.assertEqual(10, len(batches))
        self.assertEqual(1, len(batches[0]))

        # batch_size = 4, expect 3 batches of 4, 4, 2
        batches = list(batched(pdf, 4))
        self.assertEqual(3, len(batches))
        self.assertEqual(4, len(batches[0]))
        self.assertEqual(4, len(batches[1]))
        self.assertEqual(2, len(batches[2]))

        # batch_size = 10, expect 1 batch of 10
        batches = list(batched(pdf, 10))
        self.assertEqual(1, len(batches))
        self.assertEqual(10, len(batches[0]))

        # batch_size > len(pdf), expect 1 batch of 10
        batches = list(batched(pdf, 1000))
        self.assertEqual(1, len(batches))
        self.assertEqual(10, len(batches[0]))


if __name__ == '__main__':
    unittest.main()


