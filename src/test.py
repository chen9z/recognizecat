#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

a=np.array([[[[[1],[2],[3]]]]])

a=np.squeeze(a)

print(a.shape)