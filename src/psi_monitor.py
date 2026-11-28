import sys
import runpy

if __name__ == "__main__":
    runpy.run_module("src.mp249.psi_monitor", run_name="__main__")
else:
    from src.mp249 import psi_monitor as implementation
    sys.modules[__name__] = implementation