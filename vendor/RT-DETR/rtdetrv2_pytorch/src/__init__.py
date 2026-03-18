"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os

# for register purpose
# Import mode "minimal" avoids pulling training/data dependencies for export tools.
_import_mode = os.getenv('RTDETR_IMPORT_MODE', '').strip().lower()

from . import nn
from . import zoo

if _import_mode != 'minimal':
    from . import optim
    from . import data
