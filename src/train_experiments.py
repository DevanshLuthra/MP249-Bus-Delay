import sys
import runpy

if __name__ == "__main__":
    runpy.run_module("src.mp249.train_experiments", run_name="__main__")
else:
    from src.mp249 import train_experiments as implementation
    sys.modules[__name__] = implementation