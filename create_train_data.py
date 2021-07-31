import glob
import os

import click
import tqdm

from pdf_struct import loader


def process_hocr(in_path, out_path):
    with open(in_path) as fin:
        html_doc = fin.read()
    span_boxes_lst = loader.hocr.parse_hocr(html_doc)

    if len(span_boxes_lst) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    with open(out_path, 'w') as fout:
        for i, span_boxes in enumerate(span_boxes_lst):
            for span in span_boxes:
                fout.write(f'{i:>05d} {span.text}\t0\t\n')


def process_pdf(in_path, out_path):
    with open(in_path, 'rb') as fin:
        text_boxes = list(loader.pdf.parse_pdf(fin))

    if len(text_boxes) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    text_boxes = loader.pdf.TextBox.merge_continuous_lines(
        text_boxes, space_size=4)

    with open(out_path, 'w') as fout:
        for tb in text_boxes:
            fout.write(f'{tb.text}\t0\t\n')


def process_text(in_path, out_path):
    with open(in_path) as fin:
        text_lines = loader.text.TextLine.from_lines([line for line in fin])

    if len(text_lines) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    with open(out_path, 'w') as fout:
        for line in text_lines:
            fout.write(f'{line.text}\t0\t\n')


@click.command()
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
@click.argument('indir', type=click.Path(exists=True))
@click.argument('outdir', type=click.Path(exists=False))
def main(file_type, indir, outdir):
    paths = glob.glob(os.path.join(indir, f'*.{file_type}'))
    os.makedirs(outdir)
    for path in tqdm.tqdm(paths):
        out_filename = os.path.splitext(os.path.basename(path))[0] + '.tsv'
        if file_type == 'hocr':
            process_hocr(path, os.path.join(outdir, out_filename))
        elif file_type == 'pdf':
            process_pdf(path, os.path.join(outdir, out_filename))
        elif file_type == 'txt':
            process_text(path, os.path.join(outdir, out_filename))
        else:
            assert not 'Should not get here'


if __name__ == '__main__':
    main()
