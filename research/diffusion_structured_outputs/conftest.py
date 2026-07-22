# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Put the POC package dir on sys.path so tests can import the flat modules."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
