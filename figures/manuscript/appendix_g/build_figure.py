"""Rebuild the final vector asset from its saved base PDF and label overlay.

This reconstructs the manuscript-side label correction, not the experiment.
The original 500-repetition numerical output and original chart generator
were not found. No rejection counts are inferred from rounded bar labels.
"""
from pathlib import Path
import os
import hashlib
import re
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('VRT_FIGURES_OUT', str(ROOT.parents[2] / 'results' / 'figures'))) / 'appendix_g'


def main():
    source = ROOT / 'source_before.pdf'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '549701935c20d8438ae1c78e88faa17994789100401d994490a5c1aefaee6e21'
    reader = PdfReader(source)
    page = reader.pages[0]
    content = page.get_contents().get_data()
    # Preserve the first-line Concentration label and remove its old second line.
    content, n = re.subn(rb'(\(Concentration\)Tj\n).*?\nET', rb'\1ET', content, count=1, flags=re.S)
    assert n == 1
    # The original chart accidentally printed a literal backslash.
    old = b'(Uniform vs.\\\\ vMF)Tj'
    assert content.count(old) == 1
    content = content.replace(old, b'(Uniform vs. vMF)Tj')
    # Recenter that label after deleting the backslash, exactly as in the final PDF.
    assert content.count(b'201.595965 121.559373 Tm') == 1
    content = content.replace(b'201.595965 121.559373 Tm', b'202.540965 121.559373 Tm')
    stream = DecodedStreamObject()
    stream.set_data(content)
    page[NameObject('/Contents')] = stream
    page.merge_page(PdfReader(ROOT / 'label_overlay.pdf').pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'fig_ref_asymmetry.pdf'
    with path.open('wb') as f:
        writer.write(f)
    print(path.name + ': preserved original bars; reconstructed final label corrections.')


if __name__ == '__main__':
    main()
