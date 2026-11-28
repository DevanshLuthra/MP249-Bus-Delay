import sys
from src.mp249 import api as implementation

sys.modules[__name__] = implementation
