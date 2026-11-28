import sys
import runpy

if __name__ == "__main__":
    runpy.run_module("src.mp249.audit_data", run_name="__main__")
else:
    from src.mp249 import audit_data as implementation
    sys.modules[__name__] = implementation