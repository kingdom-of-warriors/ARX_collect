import os
import sys
import ctypes


def setup_loader(root):
    py = sys.version_info
    if py < (3, 8) or py >= (3, 13):
        raise RuntimeError(f"Unsupported Python version: {py.major}.{py.minor}. "
                           "Requires Python 3.8–3.12.")

    base_dir = os.path.join(root, 'msg', f'{py.major}.{py.minor}')
    lib_dir = os.path.join(base_dir, 'lib')

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    for filename in sorted(os.listdir(lib_dir)):
        if filename.endswith(".so"):
            full_path = os.path.join(lib_dir, filename)
            try:
                ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"Failed to load: {filename}\n Reason: {e}")
