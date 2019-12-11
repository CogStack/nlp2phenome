import sklearn
from os import listdir
from os.path import isfile, join
from nlp_to_phenome import EDIRDoc, EDIRAnn
import reportreader as rr
import re
import utils
import logging
import sys


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
                m = re.match(r'Verified\_([^\(]+)(\(.*\)){0,1}', mc.attrib['id'])
                if m is None:
                    m = re.match(r'(Irrelevant_label)', mc.attrib['id'])
                if m is not None:
                    cls = m.group(1)
                    if no_context and cls != 'Irrelevant_label':
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
    print('IAA is %s' % iaa)
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


def get_what_is_changing(ann_folder, text_folder, output_file):
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
        d = eHostAnnDoc(join(ann_folder, f))
        anns = d.get_ess_entities(no_context=True)
        text = utils.read_text_file_as_string(join(text_folder, f[0:-14]))
        sents = rr.get_sentences_as_anns(nlp, text)
        for ann in anns:
            for s in sents:
                if ann.overlap(s):
                    abss = rr.AbstractedSentence(1)
                    abss.text = s.str
                    result = abss.get_abstaction_by_pos(abss.locate_pos(ann.str), nlp)
                    type = ann.label
                    if type not in type2abstractions:
                        type2abstractions[type] = []
                    type2abstractions[type].append(result.to_dict())
    logging.debug(type2abstractions)
    utils.save_json_array(type2abstractions, output_file)


def compute_iaa():
    folder_lia = "lia"
    folder_rob = "rob"
    folder_nadia = "nadia"
    ehost_iaa_compute(folder_lia, folder_rob, no_context=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('the syntax is [python ann_utils.py ann_folder, text_folder, result_file]')
    else:
        get_what_is_changing(sys.argv[1], sys.argv[2], sys.argv[3])