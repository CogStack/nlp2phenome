import xml.etree.ElementTree as ET
import datetime
import logging


class AnnConverter(object):

    @staticmethod
    def get_semehr_ann_label(ann):
        str_context = ''
        if ann.negation != 'Affirmed':
            str_context += ann.negation + '_'
        if ann.temporality != 'Recent':
            str_context += ann.temporality + '_'
        if ann.experiencer != 'Patient':
            str_context += ann.experiencer + '_'
        return '%s%s' % (str_context, ann.minor_type)

    @staticmethod
    def to_eHOST(doc_key, anns, file_pattern='%s.txt', id_pattern='smehr-%s-%s'):
        elem_annotations = ET.Element("annotations")
        elem_annotations.set('textSource', file_pattern % doc_key)
        idx = 0
        for d in anns:
            ann = d['ann']
            idx += 1
            mention_id = id_pattern % (doc_key, idx)
            elem_ann = ET.SubElement(elem_annotations, "annotation")
            elem_mention = ET.SubElement(elem_ann, "mention")
            elem_mention.set('id', mention_id)
            elem_annotator = ET.SubElement(elem_ann, "annotator")
            elem_annotator.set('id', 'semehr')
            elem_annotator.text = 'semehr'
            elem_span = ET.SubElement(elem_ann, "span")
            elem_span.set('start', '%s' % ann.start)
            elem_span.set('end', '%s' % ann.end)
            elem_spanText = ET.SubElement(elem_ann, "spannedText")
            elem_spanText.text = ann.str
            elem_date = ET.SubElement(elem_ann, "creationDate")
            elem_date.text = datetime.datetime.now().strftime("%a %B %d %X %Z %Y")
            #
            elem_class = ET.SubElement(elem_annotations, "classMention")
            elem_class.set('id', mention_id)
            elem_mention_class = ET.SubElement(elem_class, "mentionClass")
            elem_mention_class.set('id', AnnConverter.get_semehr_ann_label(ann))
            elem_mention_class.text = ann.str
        tree = ET.ElementTree(elem_annotations)
        return ET.tostring(elem_annotations, encoding='utf8', method='xml')


if __name__ == "__main__":
    pass