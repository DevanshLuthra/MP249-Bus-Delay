import sys
import runpy

if __name__ == "__main__":
    runpy.run_module("src.mp249.register_models", run_name="__main__")
else:
    from src.mp249 import register_models as implementation
    sys.modules[__name__] = implementation