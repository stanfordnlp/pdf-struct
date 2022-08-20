# pdf-struct: Logical structure analysis for visually structured documents

This is a tool for extracting fine-grained logical structures (such as boundaries and their hierarchies) from visually structured documents (VSDs) such as PDFs.
pdf-struct is easily customizable to different types of VSDs and it significantly outperformed baselines in identifying different structures in VSDs.
For example, our system obtained a paragraph boundary detection F1 score of 0.953 which is significantly better than a popular PDF-to-text tool with an F1 score of 0.739.
Please note that current pdf-struct has several limitations:

* It is intended for single-column documents. It does not suport multi-column documents.
* Published models are trained on contracts. It may work on general documents, but it has not been tested. Nevertheless, you can train your own model using a corpus of your choice.

Details of pdf-struct can be found in our [paper](https://aclanthology.org/2021.nllp-1.15/) that was published in "Natural Legal Language Processing Workshop 2021".
You can find the dataset for reproducing the paper [here](https://stanfordnlp.github.io/pdf-struct-dataset/).

## Basic Usage

This program runs on Python 3 (tested on 3.8.5).
Install pdf-struct:

```
pip install pdf-struct
```

### CLI

```
pdf-struct predict --model PDFContractEnFeatureExtractor ${PATH_TO_PDF_FILE}
```

You may choose a pretrained model from https://github.com/stanfordnlp/pdf-struct-models .
Please refer `pdf-struct predict --help` for full options.

### Python Interface

`pdf-struct` provides a Python interface for inline prediction, too:

```python
import pdf_struct


pdf_struct.predict(
  format='paragraphs',
  in_path=path_to_pdf_file,
  model='PDFContractEnFeatureExtractor'
)
```

You can refer `pdf-struct predict --help` for the options, as it is basically what is used internally by CLI.

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

You can easily implement your own feature extractor.
All feature extractors inherit from `pdf_struct.core.feature_extractor.BaseFeatureExtractor`.
A child class of `BaseFeatureExtractor` has to implement feature extracting functions.
A feature extractor class will be instantiated for each document and each feature extracting function will be called for each pair of consecutive lines of the input document.

```
from typing import List, Optional

import numpy as np

from pdf_struct.core.feature_extractor import BaseFeatureExtractor, \
    single_input_feature, pairwise_feature
from pdf_struct.loader.pdf import TextBox
from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct.core.export import to_tree
from pdf_struct.core.predictor import train_classifiers, \
    predict_with_classifiers


class MinimalPDFFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, text_boxes: List[TextBox]):
        '''This class is instantiated for each document.
        The constructor receives text_boxes which is basically all the
        lines of the input document.
        You can calculate document-specific global constants here to use
        in the actual feature extraction.

        text_boxes is list of objects all of which inherit from
        `pdf_struct.core.document.TextBlock`. This will be
        `pdf_struct.loader.pdf.TextBox` if you choose `pdf` in `pdf-struct train`
        or `pdf_struct.loader.text.TextLine` if you choose `text`.
        '''
        bboxes = np.array(
            [tb.bbox for tb in text_boxes])
        page_top = bboxes[:, 3].max()
        page_bottom = bboxes[:, 1].min()
        self.header_thresh = \
            page_top - 0.15 * (page_top - page_bottom)

    @single_input_feature([1])
    def header_region(self, tb: Optional[TextBox]):
        '''A member function with `@single_input_feature` will be called
        for each pair of consecutive text blocks. For each pair, such function
        will be applied to text blocks whose indices are specified in the argument.
        `1` means the line means the first of the pair and `2` means the latter
        `0` is the line before `1` and `3` means the line after `2`.
        The function should return `bool`, `int` or `float`. It can also return
        dict (keys will be appended to the function name to create feature names)
        or list (numbers will be automatically appended).
        tb can be `None` when it is outside the document region (specifying `3`
        will results in `None` towards the end of the document).

        Here, we are classifying whether the first line is in a header region
        as this can be a strong clue when determining the relationship between
        the pair.
        '''
        return bool(tb.bbox[3] > self.header_thresh)

    @pairwise_feature([(0, 1), (1, 2)])
    def page_change(self, tb1: Optional[TextBox], tb2: Optional[TextBox]):
        '''Same as `@single_input_feature` but works on pair of text blocks
        as specified in its argument.
        '''
        if tb1 is None or tb2 is None:
           return True
        return tb1.page != tb2.page



annos = transition_labels.load_annos('./path-to-anno-dir')

FILE_TYPE = 'pdf'
documents = loader.modules[FILE_TYPE].load_from_directory('./path-to-raw-file-dir', annos)
assert len(documents) > 0

documents = [MinimalPDFFeatureExtractor.append_features_to_document(document)
             for document in documents]

clf, clf_ptr = train_classifiers(documents)

# Now make predictions
document = loader.modules[FILE_TYPE].load_document('./path-to-pdf.pdf', None, None)
document = MinimalPDFFeatureExtractor.append_features_to_document(document)

pred = predict_with_classifiers(clf, clf_ptr, [document])[0]
print(to_tree(pred))
```

Examples of feature extractors can be found in `pdf_struct.feature_extractors`.
You can also inhert from any of the existing feature extractors so that you do not need to copy-and-paste the whole class.

## Citing

If you used our work for your academic publication, please cite our work:

```
@inproceedings{koreeda-manning-2021-capturing,
    title = "Capturing Logical Structure of Visually Structured Documents with Multimodal Transition Parser",
    author = "Koreeda, Yuta  and
      Manning, Christopher",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nllp-1.15",
    doi = "10.18653/v1/2021.nllp-1.15",
    pages = "144--154"
}
```
