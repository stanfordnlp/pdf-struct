import glob
import os
from itertools import islice, zip_longest

import click


_LABEL_MAPPER = {
    0: 'c',
    1: 's',
    2: 'd',
    3: 'r',
    4: 'e'
}


def extract_footprint(text, limit):
    return ''.join(islice((c for c in text if 48 <= ord(c) <= 122), limit))


@click.command()
@click.argument('old', type=click.Path(exists=True))
@click.argument('new', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
def main(old, new, output):
    os.makedirs(output)
    old_data = dict()
    for path in glob.glob(os.path.join(old, '*.tsv')):
        d = []
        is_annotated = False
        with open(path) as fin:
            for line in fin:
                line = line.rstrip('\n').split('\t')
                if len(line) < 2 or line[1].strip() == '':
                    label = None
                else:
                    is_annotated = True
                    label =_LABEL_MAPPER[int(line[1].strip())]
                d.append((line[0], label))
        if is_annotated:
            old_data[os.path.basename(path)] = d
    print(f'Loaded {len(old_data)} with annotation')
    for path in glob.glob(os.path.join(new, '*.tsv')):
        basename = os.path.basename(path)
        if basename not in old_data:
            continue
        buf = []
        with open(path) as fin:
            for line, (text_old, label) in zip_longest(fin, old_data[basename], fillvalue=(None, None)):
                text = line.rstrip('\n').split('\t')[0]
                if label is None:
                    buf.append(f'{text}\t0\t')
                else:
                    text_footprint = extract_footprint(text, 40)
                    text_old_footprint = extract_footprint(text_old, 40)
                    if text_footprint != text_old_footprint:
                        print(
                            f'Footprint mismatch in file {path}. old: {text_old} '
                            f'vs. new: {text}')
                        break
                    buf.append(f'{text}\t0\t{label}')
            else:
                # buffer it in case there is footprint mismtach
                with open(os.path.join(output, basename), 'w') as fout:
                    fout.write('\n'.join(buf))


if __name__ == '__main__':
    main()
