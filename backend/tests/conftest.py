import os
import tempfile

# Point storage at a throwaway folder before any app module is imported, so tests never touch real data.
os.environ.setdefault("STORAGE_DIR", tempfile.mkdtemp(prefix="swing-test-"))
os.environ.setdefault("RUN_SCHEDULER", "0")
