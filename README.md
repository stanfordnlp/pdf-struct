# pdf-struct: Logical structure analysis for visually structured documents

This is a tool for extracting fine-grained logical structures (such as boundaries and their hierarchies) from visually structured documents (VSDs) such as PDFs.
pdf-struct is easily customizable to different types of VSDs and it significantly outperformed baselines in identifying different structures in VSDs.
For example, our system obtained a paragraph boundary detection F1 score of 0.953 which is significantly better than a popular PDF-to-text tool with an F1 score of 0.739.
Please note that current pdf-struct has several limitations:

* It is intended for single-column documents. It does not suport multi-column documents.
* Published models are trained on contracts. It may work on general documents, but it has not been tested. Nevertheless, you can train your own model using a corpus of your choice.

Details of pdf-struct can be found in our [paper](https://arxiv.org/abs/2105.00150) that was published in "Natural Legal Language Processing Workshop 2021".
You can find the dataset for reproducing the paper [here](https://stanfordnlp.github.io/pdf-struct-dataset/).

## Basic Usage

This program runs on Python 3 (tested on 3.8.5).
Install pdf-struct:

```
pip install pdf-struct
```

```
pdf-struct predict --model PDFContractEnFeatureExtractor ${PATH_TO_PDF_FILE}
```

You may choose a pretrained model from https://github.com/stanfordnlp/pdf-struct-models

## Advanced Usage

This section explains the way to create your own dataset and to train your own models.

### Prerequisite

To install dependencies, run:

```bash
pip install -r requirements.txt
```

### Getting data ready

First, place your raw documents in a directory of your choice.
They must have following extensions:

* `*.pdf`: PDF files with embedded text. PDF without embedded text (i.e. those that require OCR) or two columns PDF is not supported.
* `*.txt`: Plain text files that are visually structured with spaces/line breaks.
* `*.hocr`: HOCR files.

You may handle HTML files by turning them into PDF files:

```bash
find my_input_directory/ -type f | \
  grep -P 'html$|htm$|HTML$|HTM$' | \
  while read f; do \
    chrome --headless --disable-gpu --print-to-pdf-no-header --print-to-pdf="data/raw/`basename $f`.pdf" "$f"; \
  done
```


### Creating TSV files for annotation

Create TSV file for annotation. 

```bash
pdf-struct init-dataset ${FILE_TYPE} ${RAW_DOCUMENTS_DIR} ${OUTPUT_DIR}
```

where `${FILE_TYPE}` should be one of `pdf`, `txt` or `hocr`.

This will output tsv files to `${OUTPUT_DIR}`.

### Annotating TSV files

Annotate TSV files that were geenerated with `init-dataset` command.

Each line of TSV file is organized as following:

```
text<tab>pointer<tab>label
```

`text` is extracted text from the input document. It should roughly correspond to a line in the document.

`label` (default empty) denotes the transition relationship between that line and the next line.
It should be one of following:

* c(continuous): Next line is part of a same paragraph
* a(ddress): Next line is part of a same paragraph BUT the line break is meaningful. This is intended to be used for things like addresses.
* b(lock): Next line is a start of a new paragraph BUT in within the same "block"
* s(ame level): Next line is a start of a new block (thus a new paragraph)
* d(rop): Next line is a start of a new block (thus a new paragraph) that is a child of the current block
* e(limine): The current line should be deleted
* x (excluded): The current line should be excluded both from training and evaluation
    - In our experiments, we removed things like temporal two column lines, signatures, titles etc.

In the annotation, we introduced a concept `block`. This is intended for a case where we want to distinguish listings and paragraphs.
e.g.,

```
Each party must:

    1. Blah blah blah ....
    blah blah blah....
      Blah blah blah....
    blah blah blah....

    2. Blah blah blah...
```

Here, a new paragraph within `1.` at the fifth line is definately meaningful and it should not be treated in the same way as the start of `2.` at the eighth line.
We say that relationship between the forth and fith lines (i.e. label for the forth line) is `b`.

That being said, we currently treat b and s label in the same way.
In fact some other labels are merged in the training/evaluation:

* `c` and `a` will be merged
* `b` and `s` will be merged
* `c`, `b`, `d` and `s` with a pointer is merged

`pointer` (default 0) is introduced when the hierarchy goes up.
It should be used along with `c`, `b`, `d` or `s`.
We use pointer along with different labels, because we have some oocasions where we see rise in hierarchy AND the line being a continous paragraph or a different paragraph.

e.g.,

```
Blah blah blah...:<tab>0<tab>d
  a. Blah blah blah...<tab>0<tab>s
  b. Blah blah blah...<tab>-1<tab>s
Blah blah blah...:<tab>0<tab>d
  1. Blah blah blah...<tab>0<tab>d
    a) Blah blah blah...<tab>0<tab>c
     blah blah blah...<tab>0<tab>s
    b) Blah blah blah...<tab>5<tab>c
    but this does not include ...<tab>5<tab>s
                       PAGE 1/2<tab>0<tab>e
  2. Blah blah blah...<tab>0<tab>d
```

As you can see, eighth line use a pointer along with `c` because the nineth line is actually a continous paragraph from the fifth line.
Pointers are 1-indexed (starts from 1) and 0 denotes no pointer.
A pointer can be set to `-1` to return to the most upper hierarchy.
The last line should be annotated with pointer `-1` and label `s` (though it is ignored internally).
  
### Evaluating models

You can run experiments with following command:

```bash
pdf-struct evaluate ${FILE_TYPE} ${FEATURE_EXTRACTOR_NAME} ${RAW_DOCUMENTS_DIR} ${ANNOTATED_DOCUMENTS_DIR}
```

Refer `pdf-struct evaluate --help` for the list of the feature extractors.
This will run k-folds cross validation over the data.

### Training models

You can train a new model on your dataset.

```bash
pdf-struct train ${FILE_TYPE} ${FEATURE_EXTRACTOR_NAME} ${RAW_DOCUMENTS_DIR} ${ANNOTATED_DOCUMENTS_DIR} ${MODEL_OUTPUT_PATH}
```

You can then feed `${MODEL_OUTPUT_PATH}` to `--path` option of `pdf-struct predict`.

## Customizing feature extractor

Coming soon!
