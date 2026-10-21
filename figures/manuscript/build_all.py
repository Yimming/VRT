"""Build the four final manuscript figures from any working directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for folder, script in [('appendix_c', 'plot_official_paper.py'),
                       ('appendix_f', 'build_figure.py'),
                       ('appendix_g', 'build_figure.py'),
                       ('appendix_i', 'plot_horizontal.py')]:
    subprocess.run([sys.executable, str(ROOT / folder / script)], check=True)
