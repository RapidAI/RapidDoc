"""Summarize development-only Windows stack samples; not a speed benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inclusive, leaves, modules = Counter(), Counter(), Counter()
    total = native = 0
    categories, serializer_callers = Counter(), Counter()
    for line in args.source.read_text(encoding='utf8').splitlines():
        count, *frames = line.split('\t')
        count = int(count)
        total += count
        modules[frames[0].split('!')[0]] += count
        if not any('_native!' in frame for frame in frames):
            continue
        native += count
        leaves[frames[0]] += count
        inclusive.update({frame: count for frame in set(frames)})
        for category, marker in [('value_serialization', 'serde_json5value3ser'),
                                 ('json_tree_clone', 'clone_subtree'),
                                 ('aligned_allocation', '!mi_malloc_aligned'),
                                 ('inside_pdfium_library', 'rapiddoc_pdfium!'),
                                 ('native_prediction', 'native_pdf_document13predict_range')]:
            if any(marker in frame for frame in frames):
                categories[category] += count
        for i, frame in enumerate(frames):
            if 'serde_json5value3ser' in frame:
                caller = next((f for f in frames[i+1:] if '_8rapiddoc10processing' in f), '<other>')
                serializer_callers[caller] += count
                break

    def rows(counter):
        return [dict(function=f, samples=n, native_sample_percent=round(n*100/native, 2))
                for f, n in counter.most_common(60)] if native else []

    args.output.write_text(json.dumps(dict(
        source=str(args.source), total_samples=total, native_samples=native,
        inclusive=rows(inclusive), leaves=rows(leaves), leaf_modules=dict(modules),
        categories=rows(categories), value_serializer_callers=rows(serializer_callers),
        notes=['Main-thread wall-clock sampling includes waits and profiler overhead; not CPU percentages.',
               'Inclusive rows overlap; never sum them. Recursive frames count once per sample.',
               'Export-only names (e.g. PyInit__native) are insufficient for function diagnosis.',
               'Compare speed using separate uninstrumented processes.']), indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
