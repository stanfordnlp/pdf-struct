import operator
from functools import reduce


def merge_continuous_lines(text_boxes, threshold=0.5, space_size=4):
    # FIXME: Ugly workaround for circular import
    # fix the general project structures
    from pdf_struct.pdf.parser import TextBox
    from pdf_struct.hocr.parser import SpanBox

    # text_boxes must implement .bbox where bbox is
    # [x_left, y_bottom, x_right, y_top] in points with left bottom being [0, 0, 0, 0]
    assert len(set(map(type, text_boxes))) == 1
    if isinstance(text_boxes[0], TextBox):
        box_type = 'TextBox'
    elif isinstance(text_boxes[0], SpanBox):
        box_type = 'SpanBox'
    else:
        assert not 'text_boxes must be TextBox or SpanBox'
    if len(text_boxes) <= 1:
        return text_boxes
    text_boxes = sorted(
        text_boxes,
        key=lambda b: (b.page if isinstance(b, TextBox) else 0, -b.bbox[1], b.bbox[0]))
    merged_text_boxes = []
    i = 0
    while i < (len(text_boxes) - 1):
        tbi = text_boxes[i]
        # aggregate text boxes in same line then merge
        same_line_boxes = [tbi]
        for j in range(i + 1, len(text_boxes)):
            tbj = text_boxes[j]
            if isinstance(tbi, TextBox) and tbi.page != tbj.page:
                break
            # text_boxes[j]'s y_bottom is always lower than text_boxes[i]'s
            span = max(tbi.bbox[3], tbj.bbox[3]) - tbj.bbox[1]
            overlap = min(tbi.bbox[3], tbj.bbox[3]) - tbi.bbox[1]
            if overlap / span > threshold:
                same_line_boxes.append(tbj)
                continue
            else:
                # stop scanning for same line for efficiency
                break
        if len(same_line_boxes) > 1:
            # sort left to right
            same_line_boxes = sorted(same_line_boxes, key=lambda b: b.bbox[0])
            text = same_line_boxes[0].text.strip('\n')
            bbox = same_line_boxes[0].bbox
            for tbk in same_line_boxes[1:]:
                spaces = max(tbk.bbox[0] - bbox[2], 0)
                text += int(spaces // space_size) * ' '
                text += tbk.text.strip('\n')
                bbox = [
                    bbox[0],
                    min(bbox[1], tbk.bbox[1]),
                    max(bbox[2], tbk.bbox[2]),
                    max(bbox[3], tbk.bbox[3])
                ]
            blocks = reduce(operator.or_, (b.blocks for b in same_line_boxes))
            if box_type == 'TextBox':
                merged_text_boxes.append(TextBox(text, bbox, tbi.page, blocks))
            else:
                merged_text_boxes.append(SpanBox(text, bbox, blocks, tbi.cell_size))
        else:
            merged_text_boxes.append(tbi)
        i = j
    # if len(same_line_boxes) == 1 in last loop, text_boxes[-1] will be missing
    if len(same_line_boxes) == 1:
        merged_text_boxes.append(text_boxes[-1])
    return merged_text_boxes
