import glob
import os

import click
import tqdm

from pdf_struct.pdf import parse_pdf, merge_continuous_lines
from pdf_struct.text import TextLine


def process_pdf(in_path, out_path):
    with open(in_path, 'rb') as fin:
        text_boxes = merge_continuous_lines(list(parse_pdf(fin)))

    with open(out_path, 'w') as fout:
        for tb in text_boxes:
            text = tb.text.replace('\n', '').replace('\t', ' ')
            fout.write(f'{text}\t\n')


def process_text(in_path, out_path):
    with open(in_path) as fin:
        text_lines = TextLine.from_lines([line for line in fin])
    with open(out_path, 'w') as fout:
        for line in text_lines:
            fout.write(line.text.replace('\t', ' ').rstrip('\n') + '\t\n')


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
def main(file_type):
    paths = glob.glob(os.path.join('data', 'raw', f'*.{file_type}'))
    out_dir = os.path.join('data', f'anno_{file_type}')
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    for path in tqdm.tqdm(paths):
        out_filename = os.path.splitext(os.path.basename(path))[0] + '.tsv'
        if file_type == 'pdf':
            process_pdf(path, os.path.join(out_dir, out_filename))
        elif file_type == 'txt':
            process_text(path, os.path.join(out_dir, out_filename))
        else:
            assert not 'Should not get here'


if __name__ == '__main__':
    main()
