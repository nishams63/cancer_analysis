import sys
from pathlib import Path

# Add src and scripts to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
scripts_dir = root_dir / "scripts"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
