# Ground Truth Evaluation Assets

This directory stores the reference transcripts used by `deepseek-ocr/scripts/benchmark_suite.py`.

## Expected Layout

```
ground_truth/
├── sample_invoice.pdf        # lives in ../ (eval set)
├── sample_invoice.txt        # this file (UTF-8) contains the exact transcript
├── datasheet_page1.png       # lives in ../
└── datasheet_page1.txt       # transcript for the PNG
```

* Each PDF/image in `deepseek-ocr/data/eval` must have a matching `.txt` file in this directory sharing the same stem.
* Text files should be UTF-8 encoded, normalized (no exotic whitespace), and free of OCR notes.

## Sample Entry

`sample_invoice.txt`

```
Contoso Corp.
Invoice #0000
Balance Due: 0.00
```

You can remove this sample once you add real ground-truth assets sourced from your research corpus.
