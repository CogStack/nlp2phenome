import xml.etree.ElementTree as ET
import datetime
from os import listdir
from os.path import isfile, join
import utils
import csv


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
    def load_ann_file(f, do_multi=True):
        tree = ET.parse(f)
        doc = tree.getroot()
        ann2label = {}
        ann2freq = {}
        for ann in doc.findall("annotation"):
            m_id = ann.find("mention").attrib["id"]
            cm = doc.find('.//classMention[@id="' + m_id + '"]')
            cls =cm.find('mentionClass').attrib["id"]
            m_span = ann.find("span").attrib
            annid = 'm-%s-%s' % (m_span['start'], m_span['end'])
            m_text = ann.find("spannedText").text
            freq = 0
            if annid not in ann2freq:
                ann2freq[annid] = 1
            else:
                if do_multi:
                    ann2freq[annid] += 1
            annid_freq = '%s:%s' % (annid, ann2freq[annid])
            ann2label[annid_freq] = {"text": m_text, "class": cls}
        return ann2label

    @staticmethod
    def convert_csv_annotations(csv_file, text_folder, ann_folder, mapping_file, annotated_anns_file,
                                id_pattern='%s-%s', ann_file_pattern='%s.txt.knowtator.xml'):
        with open(csv_file, newline='') as cf:
            reader = csv.DictReader(cf)
            label2concepts = {}
            d2annotated_anns = {}
            for r in reader:
                d2annotated_anns[r['doc_id'] + ".txt"] = [{'s': r['start'], 'e': r['end']}]
                if r['Skip Document'] != 'Yes':
                    utils.save_string(r['text'], join(text_folder, r['doc_id'] + ".txt"))
                    elem_annotations = ET.Element("annotations")
                    elem_annotations.set('textSource', r['doc_id'])
                    mention_id = id_pattern % (r['doc_id'], 0)
                    if r['Correct'] == 'Yes' and r['Negation'] == 'NOT Negated':
                        AnnConverter.create_elem_ann(elem_annotations, mention_id,
                                                     r['start'], r['end'], r['string_orig'], r['icd10-ch'])
                    xml = ET.tostring(elem_annotations, encoding='unicode', method='xml')
                    utils.save_string(xml, join(ann_folder, ann_file_pattern % r['doc_id']))
                    if r['icd10-ch'] not in label2concepts:
                        label2concepts[r['icd10-ch']] = []
                    if r['cui'] not in label2concepts[r['icd10-ch']]:
                        label2concepts[r['icd10-ch']].append(r['cui'])
            utils.save_json_array(label2concepts, mapping_file)
            utils.save_json_array(d2annotated_anns, annotated_anns_file)

    @staticmethod
    def populate_inter_annotator_results(ann_folder_1, ann_folder_2, output_file, missing_file,
                                         correct_labels = ["VERIFIED_CORRECT"]):
        ann_files = [f for f in listdir(ann_folder_1) if isfile(join(ann_folder_1, f))]
        all_mentions = 0
        missed = []
        mismatched = []
        for f in ann_files:
            ann1 = AnnConverter.load_ann_file(join(ann_folder_1, f))
            ann2 = AnnConverter.load_ann_file(join(ann_folder_2, f))
            all_mentions += len(ann1)
            for ann in ann1:
                if ann not in ann2:
                    missed.append('%s\t%s\t%s' % (ann, ann1[ann]['text'], ann1[ann]['class']))
                elif ann2[ann]['class'] != ann1[ann]['class'] and ann1[ann]['class'] not in correct_labels:
                    mismatched.append('%s\t%s\t%s\t%s\t%s' % (f, ann, ann1[ann]['text'], ann1[ann]['class'], ann2[ann]['class']))
        print('\n'.join(mismatched))
        print(len(missed), all_mentions)
        utils.save_string('\n'.join(mismatched), output_file)
        utils.save_string('\n'.join(missed), missing_file)
   
    @staticmethod
    def calculate_IAA(ann_folder_1, ann_folder_2, output_file):
        from sklearn.metrics import cohen_kappa_score
        ann_files = [f for f in listdir(ann_folder_1) if isfile(join(ann_folder_1, f))]
        ann1_annotations = {}
        ann2_annotations = {}
        for f in ann_files:
            ann1 = AnnConverter.load_ann_file(join(ann_folder_1, f), do_multi=False)
            ann2 = AnnConverter.load_ann_file(join(ann_folder_2, f), do_multi=False)
            for ann in ann1:
                ann1_annotations['%s_%s' % (f, ann)] = ann1[ann]['class']
            for ann in ann2:
                ann2_annotations['%s_%s' % (f, ann)] = ann2[ann]['class']
        merged_anns = list(set(list(ann1_annotations.keys()) + list(ann2_annotations.keys())))
        ann1_merged = []
        ann2_merged = []
        label_missed = 'missed'
        cat2pares = {'subject': {'ann1': [], 'ann2': []}, 
                     'irrelevant': {'ann1': [], 'ann2': []},
                     'trajectory': {'ann1': [], 'ann2': []}, 
        }
        output = []
        for ann in merged_anns:
            ann1_label = label_missed if ann not in ann1_annotations else ann1_annotations[ann]
            ann2_label = label_missed if ann not in ann2_annotations else ann2_annotations[ann]
            ann1_merged.append(ann1_label)
            ann2_merged.append(ann2_label)
            if ann1_label == 'Irrelevant_label' or ann2_label == 'Irrelevant_label':
                cat2pares['irrelevant']['ann1'].append(ann1_label)
                cat2pares['irrelevant']['ann2'].append(ann2_label)
            elif ann1_label in ['Trajectory_Subject', 'General_Trajectory'] or ann2_label in ['Trajectory_Subject', 'General_Trajectory']:
                cat2pares['subject']['ann1'].append(ann1_label)
                cat2pares['subject']['ann2'].append(ann2_label)
            elif ann1_label in ['better(Trajetory)', 'worse(Trajectory)'] or ann2_label in ['better(Trajetory)', 'worse(Trajectory)']:
                cat2pares['trajectory']['ann1'].append(ann1_label)
                cat2pares['trajectory']['ann2'].append(ann2_label)
            output.append('%s\t%s\t%s' % (ann, ann1_label, ann2_label))

        print('kappa score: [%s]', cohen_kappa_score(ann1_merged, ann2_merged))
        for cat in cat2pares:
            print('%s kappa score: [%s]' % (cat, cohen_kappa_score(cat2pares[cat]['ann1'], cat2pares[cat]['ann2'])))
        utils.save_string('\n'.join(output), output_file)

if __name__ == "__main__":
    # AnnConverter.load_ann_file('S:/NLP/annotation_Steven/stroke_nlp/saved/Stroke_id_105.txt.knowtator.xml')
    # AnnConverter.populate_inter_annotator_results('S:/NLP/annotation_Kristiina/stroke_nlp/saved',
    # 'S:/NLP/annotation_Steven/stroke_nlp/saved', 'mismatched.tsv')
    # AnnConverter.populate_inter_annotator_results('S:/NLP/annotation_Steven/stroke_nlp/saved',
    #                                               'P:/wuh/SemEHR-working/outputs/nlp2phenome',
    #                                               'kristiina_corrections.tsv', 'steven_added.tsv')
    ann_folder = '/data/annotated_data/'
    ann_files = [f for f in listdir(ann_folder) if isfile(join(ann_folder, f))]
    for f in ann_files:
        print('processing %s...' % f)
        AnnConverter.convert_csv_annotations(join(ann_folder, f), join(ann_folder, 'corpus'), join(ann_folder, 'gold'), join(ann_folder, 'concept_mapping.json'), join(ann_folder, 'annotated_anns.json'))
