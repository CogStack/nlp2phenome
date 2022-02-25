import sklearn
import datetime
from os import listdir
from os.path import isfile, join
from nlp_to_phenome import EDIRDoc
from annotation_docs import EDIRAnn
import reportreader as rr
import re
import utils
import logging
from operator import itemgetter
import xml.etree.ElementTree as ET


class eHostGenedDoc(EDIRDoc):
    def __init__(self, file_path):
        super(eHostGenedDoc, self).__init__(file_path)

    def get_ess_entities(self):
        if self._entities is not None:
            return self._entities
        root = self._root
        entities = []
        for e in root.findall('.//classMention'):
            mcs = e.findall('./mentionClass')
            mention_id = e.attrib['id']
            if len(mcs) > 0:
                mc = mcs[0]
                cls = mc.attrib['id']
                mentions = root.findall('.//mention[@id="' + mention_id + '"]/..')
                if len(mentions) > 0:
                    span = mentions[0].findall('./span')
                    ent_start = span[0].attrib['start']
                    ent_end = span[0].attrib['end']
                    spannedText = mentions[0].findall('./spannedText')
                    str = spannedText[0].text
                    ann = EDIRAnn(str=str, start=int(ent_start), end=int(ent_end), type=cls)
                    ann.id = len(entities)
                    entities.append(ann)
        self._entities = entities
        return self._entities


class eHostAnnDoc(EDIRDoc):
    """
    a document class for ehost annotation file
    """
    def __init__(self, file_path):
        super(eHostAnnDoc, self).__init__(file_path)

    def get_ess_entities(self, no_context=False):
        if self._entities is not None:
            return self._entities
        root = self._root
        entities = []
        for e in root.findall('.//classMention'):
            mcs = e.findall('./mentionClass')
            mention_id = e.attrib['id']
            if len(mcs) > 0:
                mc = mcs[0]
                m = re.match(r'VERIFIED\_([^\(]+)', mc.attrib['id'])
                if m is None:
                    m = re.match(r'(IRRELEVANT_LABELS)', mc.attrib['id'])
                if m is None:
                    m = re.match(r'(ADDED)\_([^\(]+)', mc.attrib['id'])
                if m is not None:
                    cls = m.group(1)
                    if no_context and cls != 'IRRELEVANT_LABELS':
                        if cls.find('_') >= 0:
                            cls = cls[cls.find('_')+1:]
                    mentions = root.findall('.//mention[@id="' + mention_id + '"]/..')
                    if len(mentions) > 0:
                        span = mentions[0].findall('./span')
                        ent_start = span[0].attrib['start']
                        ent_end = span[0].attrib['end']
                        spannedText = mentions[0].findall('./spannedText')
                        str = spannedText[0].text
                        ann = EDIRAnn(str=str, start=int(ent_start), end=int(ent_end), type=cls)
                        ann.id = len(entities)
                        entities.append(ann)
        self._entities = entities
        return self._entities


def ehost_iaa_compute(folder1, folder2, no_context=False):
    """
    compute inter annotator agreement
    :param folder1:
    :param folder2:
    :param no_context:
    :return:
    """
    annotator1 = read_ehost_annotated_result(folder1, no_context=no_context)
    annotator2 = read_ehost_annotated_result(folder2, no_context=no_context)
    merged_keys = list(set(annotator1.keys()) | set(annotator2.keys()))
    y1 = []
    y2 = []
    for key in merged_keys:
        if key in annotator1 and key in annotator2:
            y1.append(annotator1[key])
            y2.append(annotator2[key])
        else:
            print('%s not matched in all' % key)
    iaa = sklearn.metrics.cohen_kappa_score(y1, y2)
    print('IAA is %s on %s' % (iaa, len(annotator1)))
    return iaa


def read_ehost_annotated_result(folder, no_context=False):
    """
    read ehost annotated documents as a dictionary object: id -> entity label
    :param folder:
    :param no_context:
    :return:
    """
    id2label = {}
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    for f in files:
        d = eHostAnnDoc(join(folder, f))
        for e in d.get_ess_entities(no_context=no_context):
            id = '%s-%s-%s' % (f, e.start, e.end)
            id2label[id] = e.label
    print(id2label)
    return id2label


def get_what_is_changing(ann_folder, text_folder, output_file, eHostAnnFile=True):
    """
    get what is getting better/worse
    :param ann_folder:
    :param text_folder:
    :param output_file:
    :return:
    """
    nlp = rr.get_nlp_instance()
    files = [f for f in listdir(ann_folder) if isfile(join(ann_folder, f))]
    type2abstractions = {}
    for f in files:        
        anns = []
        text_file = join(text_folder, f[0:-14])
        if eHostAnnFile:
            d = eHostAnnDoc(join(ann_folder, f))
            anns = d.get_ess_entities(no_context=True)
        else:
            d = eHostGenedDoc(join(ann_folder, f))
            anns = d.get_ess_entities()
        if len(anns) == 0:
            logging.info('anns is empty for [{:s}]'.format(f))
        text = utils.read_text_file_as_string(join(text_folder, f[0:-14]), encoding='cp1252')
        sents = rr.get_sentences_as_anns(nlp, text)
        for ann in anns:
            for s in sents:
                if ann.overlap(s):
                    abss = rr.AbstractedSentence(1)
                    abss.text = s.str
                    result = abss.get_abstaction_by_pos(abss.locate_pos(ann.str), nlp)
                    if result is None:
                        logging.info('%s not found in %s' % (ann.str, f))
                        continue
                    type = ann.label
                    if type not in type2abstractions:
                        type2abstractions[type] = []
                    type2abstractions[type].append(result.to_dict())
    logging.debug(type2abstractions)
    utils.save_json_array(type2abstractions, output_file)


def compute_iaa():
    folder_lia = "S:/NLP/annotation_it02/overlaps/k"
    folder_rob = "S:/NLP/annotation_it02/overlaps/s"
    folder_nadia = "nadia"
    ehost_iaa_compute(folder_lia, folder_rob, no_context=True)


def analysing_label_performance(folder, output_file):
    s2t = {}
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    for f in files:
        d = eHostAnnDoc(join(folder, f))
        for ann in d.get_ess_entities():
            s = ann.str
            if not (s in s2t):
                s2t[s] = {}
            if ann.type in s2t[s]:
                s2t[s][ann.type] = s2t[s][ann.type] + 1
            else:
                s2t[s][ann.type] = 1
    sts = sorted([(s, s2t[s]['CORRECT'] if 'CORRECT' in s2t[s] else 0, s2t[s]['IRRELEVANT_LABELS'] if 'IRRELEVANT_LABELS' in s2t[s] else 0, s2t[s]['ADDED'] if 'ADDED' in s2t[s] else 0) for s in s2t], key=itemgetter(2), reverse=True)
    s = ('\n'.join(['%s\t%s\t%s\t%s' % (t[0], t[1], t[2], t[3]) for t in sts]))
    utils.save_string(s, output_file)


def generate_gold_stand_from_validation(generated_ann_folder, validated_ann_folder, gold_standard_folder):

    files = [f for f in listdir(generated_ann_folder) if isfile(join(generated_ann_folder, f))]
    for f in files:
        logging.debug('processing: %s / %s' % (generated_ann_folder, f))
        # ignore added annotations for now 
        gd_anns = []
        gen_doc = eHostGenedDoc(join(generated_ann_folder, f))
        logging.debug('ann number: %s' % len(gen_doc.get_ess_entities()))
        val_doc = eHostAnnDoc(join(validated_ann_folder, f))
        for g in gen_doc.get_ess_entities():
            logging.debug('validation label: %s' % g.type)
            for v in val_doc.get_ess_entities():
                if g.start == v.start and g.end == v.end:
                    logging.debug('validation label: %s' % v.type)
                    if v.type == 'CORRECT':
                        gd_anns.append(g)

        elem_annotations = ET.Element("annotations")
        elem_annotations.set('textSource', f)
        idx = 0
        for ann in gd_anns:
            if ann.str.lower() == 'haematoma':
                continue
            idx += 1
            mention_id = '%s-%s' % (f, idx)
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
            if ann.str.lower() == 'haemorrhage' or ann.str.lower() == 'blood' or ann.str.lower() == 'bleed' or ann.str.lower().startswith('collection'):
                ann.type = 'bleeding'
            elem_mention_class.set('id', ann.type)
            elem_mention_class.text = ann.str
        tree = ET.ElementTree(elem_annotations)
        logging.info('gd file saved to %s - %s' % (gold_standard_folder, f))
        utils.save_string(ET.tostring(elem_annotations, encoding='utf8', method='xml'), join(gold_standard_folder, f))


def analyse_trajectory_subjects(file, output_file):
    t2subs = utils.load_json_data(file)
    t2freq = {}
    for t in t2subs:
        if t not in t2freq:
            t2freq[t] = {'subject': {}, 'root': {}}
        for sub in t2subs[t]:
            add_key_freq(t2freq[t]['subject'], ','.join(sub['subject']))
            add_key_freq(t2freq[t]['root'], sub['root'])

    s = ''
    for t in t2freq:
        freqs = t2freq[t]
        subs = sorted([(k, freqs['subject'][k]) for k in freqs['subject']], key=itemgetter(1), reverse=True)
        s += '***%s [subjects]***\n%s\n\n' % (t, freq_to_str(subs))
        roots = sorted([(k, freqs['root'][k]) for k in freqs['root']], key=itemgetter(1), reverse=True)
        s += '***%s [roots]***\n%s\n\n' % (t, freq_to_str(roots))
    logging.info(s)
    utils.save_string(s, output_file)


def freq_to_str(freq):
    return '\n'.join(['%s\t%s' % (t[0], t[1]) for t in freq])


def add_key_freq(d, key):
    if key in d:
        d[key] += 1
    else:
        d[key] = 1


def summarise_validation_results(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    t2freq = {}
    for f in files:
        gen_doc = eHostGenedDoc(join(folder, f))
        logging.debug('processing: %s / %s' % (folder, f))
        for g in gen_doc.get_ess_entities():
            logging.debug('validation label: %s' % g.type)
            if g.type not in t2freq:
                t2freq[g.type] = 0
            t2freq[g.type] += 1
    s = '\n'.join(['%s\t%s' % (t, t2freq[t]) for t in t2freq])
    logging.info(s)
    return s

            

if __name__ == "__main__":
    log_level = 'DEBUG'
    log_format = '[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s'
    logging.basicConfig(level='DEBUG', format=log_format)
    # compute_iaa()
    # analysing_label_performance('S:/NLP/annotation_it02/annotation_Steven/iteration_02/saved', 
    #                             'P:/wuh/label2performce_steve.tsv')
    # generate_gold_stand_from_validation('P:/wuh/SemEHR-working/outputs_it2/nlp2phenome',
    #                                     'S:/NLP/annotation_it02/annotation_Steven/iteration_02/saved',
    #                                     'P:/wuh/SemEHR-working/outputs_it2/gold_stand_results')
    sub_json_file = './diabetes_subs.json'
    analyse_trajectory_subjects(sub_json_file, './traject_sub_analysis_result.txt')
    # if len(sys.argv) != 4:
    #     print('the syntax is [python ann_utils.py ann_folder, text_folder, result_file]')
    # else:
    #     logging.info('working...')
    #     get_what_is_changing(sys.argv[1], sys.argv[2], sys.argv[3], eHostAnnFile=False)
    # summarise_validation_results('/data/val/it2')