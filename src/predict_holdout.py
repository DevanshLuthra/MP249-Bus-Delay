import sys
import runpy

if __name__ == "__main__":
    runpy.run_module("src.mp249.predict_holdout", run_name="__main__")
else:
    from src.mp249 import predict_holdout as implementation
    sys.modules[__name__] = implementation