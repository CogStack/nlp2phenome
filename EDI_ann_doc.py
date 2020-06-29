from annotation_docs import EDIRAnn, relocate_annotation_pos
import logging
from os.path import basename, isfile, join, split
import xml.etree.ElementTree as ET
import re


class EDIRDoc(object):
    """
    a class for reading EDIR annotation doc (XML)
    """

    def __init__(self, file_path):
        self._path = file_path
        self._root = None
        self._full_text = None
        self._word_offset_start = -1
        self._entities = None
        self.load()

    @property
    def file_path(self):
        return self._path

    def load(self):
        if not isfile(self.file_path):
            logging.debug('%s is NOT a file' % self.file_path)
            return
        tree = ET.parse(self.file_path)
        self._root = tree.getroot()
        self.get_word_offset_start()

    @property
    def get_full_text(self):
        if self._full_text is not None:
            return self._full_text
        if self._root is None:
            self.load()
        root = self._root
        d = ''
        start_offset = -1
        for p in root.findall('.//p'):
            for s in p:
                if 'proc' in s.attrib:  # and s.attrib['proc'] == 'yes':
                    for w in s:
                        id_val = int(w.attrib['id'][1:])
                        if start_offset == -1:
                            start_offset = id_val
                        offset = id_val - start_offset
                        d += ' ' * (offset - len(d)) + w.text
        self._full_text = d
        return d

    def get_word_offset_start(self):
        if self._word_offset_start >= 0:
            return self._word_offset_start
        root = self._root
        offset_start = -1
        for e in root.findall('.//p/s[@proc]/w'):
            if 'id' not in e.attrib:
                continue
            else:
                offset_start = int(e.attrib['id'][1:])
                break
        # if offset_start == -1:
        #     logging.debug('%s offset start could not be found' % self.file_path)
        self._word_offset_start = offset_start

    def get_ess_entities(self):
        if self._entities is not None:
            return self._entities
        root = self._root
        offset_start = self.get_word_offset_start()
        entities = []
        for e in root.findall('.//standoff/ents/ent'):
            if 'type' not in e.attrib:
                continue
            ent_type = e.attrib['type']
            if ent_type.startswith('label:'):
                continue
            negated = False
            if 'neg_' in ent_type:
                negated = True
                ent_type = ent_type.replace(r'neg_', '')
            str = ' '.join([part.text for part in e.findall('./parts/part')])
            ent_start = -1
            ent_end = -1
            for part in e.findall('./parts/part'):
                ent_start = int(part.attrib['sw'][1:]) - offset_start
                ent_end = ent_start + len(part.text)
            ann = EDIRAnn(str=str, start=ent_start, end=ent_end, type=ent_type)
            ann.negated = negated
            ann.id = len(entities)
            entities.append(ann)
        self._entities = entities
        return self._entities

    def relocate_anns(self, t):
        if self._entities is None:
            return
        for a in self._entities:
            s, e = relocate_annotation_pos(t, a.start, a.end, a.str)
            a.start = s
            a.end = e


class eHostGenedDoc(EDIRDoc):
    def __init__(self, file_path):
        super(eHostGenedDoc, self).__init__(file_path)

    def get_ess_entities(self):
        if self._entities is not None:
            return self._entities
        root = self._root
        entities = []
        s_e_ids = []
        for e in root.findall('.//classMention'):
            mcs = e.findall('./mentionClass')
            mention_id = e.attrib['id']
            if len(mcs) > 0:
                mc = mcs[0]
                cls = mc.attrib['id']
                cls = cls.replace('Negated_', '').replace('hypothetical_', '').replace('Other_', '').replace(
                    'historical_', '')
                mentions = root.findall('.//mention[@id="' + mention_id + '"]/..')
                if len(mentions) > 0:
                    span = mentions[0].findall('./span')
                    ent_start = span[0].attrib['start']
                    ent_end = span[0].attrib['end']

                    s_e_id = '%s-%s' % (ent_start, ent_end)
                    if s_e_id in s_e_ids:
                        continue
                    s_e_ids.append(s_e_id)

                    spannedText = mentions[0].findall('./spannedText')
                    str = spannedText[0].text
                    ann = EDIRAnn(str=str, start=int(ent_start), end=int(ent_end), type=cls)
                    ann.id = len(entities)
                    entities.append(ann)
        self._entities = entities
        return self._entities


class eHostDoc(EDIRDoc):
    def __init__(self, file_path):
        super(eHostDoc, self).__init__(file_path)

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
                m = re.match(r'Verified\_([^\(]+)(\(.*\)){0,1}', mc.attrib['id'])
                if m is not None:
                    cls = m.group(1)
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


class ConllDoc(EDIRDoc):
    """
    for Conll output from classification results
    """

    def __init__(self, file_path):
        super(ConllDoc, self).__init__(file_path)
        self._tokens = None
        self._label_white_list = None

    def set_label_white_list(self, labels):
        self._label_white_list = labels

    @property
    def conll_output(self):
        try:
            return '\n'.join([' '.join([t['t'], str(len(t['predicted_label'])), t['gold_label'],
                                        (('B-' if t['predicted_label'][-1]['ann'].start == t['offset'] else 'I-') +
                                         t['predicted_label'][-1]['label'])
                                        if len(t['predicted_label']) > 0 else 'O'])
                              for t in self.get_token_list()])
        except:
            logging.error('processing [%s] failed' % self.file_path)
            return ''

    def get_token_list(self):
        if self._tokens is not None:
            return self._tokens
        self._tokens = []
        start_offset = -1
        root = self._root
        work_ess = list(self.get_ess_entities())
        matched_ess = set()
        for p in root.findall('.//p'):
            for s in p:
                if 'proc' in s.attrib:  # and s.attrib['proc'] == 'yes':
                    for w in s:
                        id_val = int(w.attrib['id'][1:])
                        if start_offset == -1:
                            start_offset = id_val
                        offset = id_val - start_offset
                        token = {'t': w.text, 'id': w.attrib['id'], 'offset': offset,
                                 'gold_label': 'O', 'predicted_label': []}
                        for e in work_ess:
                            label = e.type.replace('neg_', '').lower().strip()
                            if self._label_white_list is not None and label not in self._label_white_list:
                                continue
                            if token['offset'] == e.start:
                                token['gold_label'] = 'B-' + label
                                matched_ess.add(e)
                            elif e.start < token['offset'] < e.end:
                                token['gold_label'] = 'I-' + label
                                matched_ess.add(e)
                        self._tokens.append(token)
        left_ess = [e for e in work_ess if e not in matched_ess
                    and e.type.replace('neg_', '') in self._label_white_list]
        if len(left_ess) > 0:
            logging.error('leftovers: [%s] at %s' % (
                '\n'.join(['%s (%s,%s)' % (a.type, a.start, a.end) for a in left_ess]), self.file_path))
        return self._tokens

    def add_predicted_labels(self, predicted_label):
        """
        append prediction result to the doc, one annotation a time
        :param predicted_label: labelled ann {'label': ..., 'ann': ann object}
        :return:
        """
        if self._label_white_list is not None and predicted_label['label'] not in self._label_white_list:
            return
        for token in self.get_token_list():
            if predicted_label['ann'].start <= token['offset'] < predicted_label['ann'].end:
                token['predicted_label'].append(predicted_label)
