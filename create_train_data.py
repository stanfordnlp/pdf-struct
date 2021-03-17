import glob
import os

import click
import tqdm

from pdf_struct.pdf import parse_pdf
from pdf_struct.text import TextLine
from pdf_struct.bbox import merge_continuous_lines
from pdf_struct.hocr import parse_hocr


def process_hocr(in_path, out_path):
    with open(in_path) as fin:
        html_doc = fin.read()
    span_boxes_lst = parse_hocr(html_doc)

    with open(out_path, 'w') as fout:
        for i, span_boxes in enumerate(span_boxes_lst):
            for span in span_boxes:
                fout.write(f'{i:>05d} {span.text}\t0\t\n')


def process_pdf(in_path, out_path):
    with open(in_path, 'rb') as fin:
        text_boxes = merge_continuous_lines(list(parse_pdf(fin)), space_size=4)

    with open(out_path, 'w') as fout:
        for tb in text_boxes:
            fout.write(f'{tb.text}\t0\t\n')


def process_text(in_path, out_path):
    with open(in_path) as fin:
        text_lines = TextLine.from_lines([line for line in fin])
    with open(out_path, 'w') as fout:
        for line in text_lines:
            fout.write(f'{line.text}\t0\t\n')


@click.command()
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
def main(file_type):
    paths = glob.glob(os.path.join('data', 'raw', f'*.{file_type}'))
    out_dir = os.path.join('data', f'anno_{file_type}')
    os.makedirs(out_dir)
    for path in tqdm.tqdm(paths):
        out_filename = os.path.splitext(os.path.basename(path))[0] + '.tsv'
        if file_type == 'hocr':
            process_hocr(path, os.path.join(out_dir, out_filename))
        elif file_type == 'pdf':
            process_pdf(path, os.path.join(out_dir, out_filename))
        elif file_type == 'txt':
            process_text(path, os.path.join(out_dir, out_filename))
        else:
            assert not 'Should not get here'


if __name__ == '__main__':
    main()
