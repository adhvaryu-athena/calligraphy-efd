"""Calligraphy & Math: Classifying typeface styles with Elliptic Fourier Descriptors.

Five-stage pipeline:
    Stage 1: corpus       — locate and validate fonts (corpus.py)
    Stage 2: outlines     — extract outer glyph contours (outlines.py)
    Stage 3: efd          — compute EFD feature vectors (efd.py)
    Stage 4: classify     — train + evaluate three classifiers (classify.py)
    Stage 5: visualize    — produce four publication figures (visualize.py)

Run the whole thing from notebooks/pipeline.ipynb.
"""
