import xml.etree.ElementTree as ET
import datetime
import csv
import utils
from os.path import join
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
            AnnConverter.create_elem_ann(elem_annotations, mention_id, ann.start, ann.end, ann.str,
                                         AnnConverter.get_semehr_ann_label(ann))
        tree = ET.ElementTree(elem_annotations)
        return ET.tostring(elem_annotations, encoding='utf8', method='xml')

    @staticmethod
    def create_elem_ann(elem_annotations, mention_id, start, end, str, class_label):
        elem_ann = ET.SubElement(elem_annotations, "annotation")
        elem_mention = ET.SubElement(elem_ann, "mention")
        elem_mention.set('id', mention_id)
        elem_annotator = ET.SubElement(elem_ann, "annotator")
        elem_annotator.set('id', 'semehr')
        elem_annotator.text = 'semehr'
        elem_span = ET.SubElement(elem_ann, "span")
        elem_span.set('start', '%s' % start)
        elem_span.set('end', '%s' % end)
        elem_spanText = ET.SubElement(elem_ann, "spannedText")
        elem_spanText.text = str
        elem_date = ET.SubElement(elem_ann, "creationDate")
        elem_date.text = datetime.datetime.now().strftime("%a %B %d %X %Z %Y")
        #
        elem_class = ET.SubElement(elem_annotations, "classMention")
        elem_class.set('id', mention_id)
        elem_mention_class = ET.SubElement(elem_class, "mentionClass")
        elem_mention_class.set('id', class_label)
        elem_mention_class.text = str
        return elem_ann

    @staticmethod
    def convert_csv_annotations(csv_file, text_folder, ann_folder,
                                id_pattern='%s-%s', ann_file_pattern='%s.knowtator.xml'):
        with open(csv_file, newline='') as cf:
            reader = csv.DictReader(cf)
            for r in reader:
                if r['Skip Document'] != 'Yes':
                    utils.save_string(r['text'], join(text_folder, r['doc_id']))
                    elem_annotations = ET.Element("annotations")
                    elem_annotations.set('textSource', r['doc_id'])
                    mention_id = id_pattern % (r['doc_id'], 0)
                    if r['Correct'] == 'Yes' and r['Negation'] == 'NOT Negated':
                        AnnConverter.create_elem_ann(elem_annotations, mention_id,
                                                     r['_start'], r['_end'], r['string_orig'], r['cui'])
                    xml = ET.tostring(elem_annotations, encoding='utf8', method='xml')
                    utils.save_string(xml, join(ann_folder, ann_file_pattern % r['doc_id']))


if __name__ == "__main__":
    pass