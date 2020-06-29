import utils
from sklearn.ensemble import RandomForestClassifier


class PhenomeDocument(object):
    """
    a document for deriving phenotypes
    """
    def __init__(self):
        pass

    @property.getter
    def sentences(self):
        pass


class PhenomeSentence(object):
    """
    a phenome sentence is essentially a container of phenotypes and other
    language related stuffs
    """
    def __init__(self):
        self._phenotypes = None
        self._text = None

    @property.getter
    def phenotype_features(self):
        return self._phenotypes

    @property.setter
    def phenotype_features(self, values):
        self._phenotypes = values

    @property.setter
    def sentence_text(self, value):
        self._text = value

    @property.getter
    def sentence_text(self):
        return self._text


class PhenomePredictor(object):
    def __init__(self):
        pass

    def learn(self):
        pass

    def load_model(self, model_path):
        pass

    def predict(self, docs):
        pass


def learn_and_predict_phenome(setting_file):
    pass