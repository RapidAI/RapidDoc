# Table Transformer

Model: microsoft/table-transformer-structure-recognition-v1.1-all (MIT),
Hugging Face revision 7587a7ef111d9dcbf8ac695f1376ab7014340a0c.

`postprocess.py` is copied from Microsoft table-transformer `src/postprocess.py`,
with the Rect import changed to a local finite-rectangle implementation.
`structure.py` contains selected unchanged functions from `src/inference.py`,
with minimal imports, local Rect and relative postprocess import. Copyright Microsoft Corporation.
Source revision: 16d124f616109746b7785f03085100f1f6247575.
Source: https://github.com/microsoft/table-transformer . See LICENSE (MIT).

The adapter adds NumPy output conversion, single-table selection, empty-grid guard,
and HTML serialization with aligned logical spans and clipped input-coordinate boxes.

RapidDoc integration additionally preserves HTML rows fully covered by rowspan,
applies the declared table confidence threshold before single-table selection,
and validates OCR batch alignment rather than silently dropping supplied text.
