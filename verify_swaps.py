import os
import sys
import subprocess

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anemia-detection-main')
    os.chdir(project_dir)
    sys.exit(subprocess.call([sys.executable, 'verify_swaps.py']))
