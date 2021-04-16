# pdf-struct: Logical structure analysis for visually structured documents

## Prerequisite

This program runs on Python 3 (tested on 3.8.5).
To install dependencies, run:

```bash
pip install -r requirements.txt
```

## How to run

Currently, this program is structured to maximize ease of conducting experiments.
**Thus, it does not contain a command to conduct inference on unlabeled data.***
This section explains the way to conduct an experiment on PDF/text/HOCR.

### Getting data ready

First, place your documents in `data/raw`. They must have following extensions:

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
python create_train_data.py ${FILE_TYPE}
```

where `${FILE_TYPE}` should be one of `pdf`, `txt` or `hocr`.

This will output tsv files to `data/anno_${FILE_TYPE}`.

### Annotating TSV files

Carry annotation by directly editing files in `data/anno_${FILE_TYPE}`.

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
  
### Training and evaluating models

You can run experiments with following command:

```bash
python evaluate_ml.py ${FILE_TYPE}
```

This will run k-folds cross validation over the data.
