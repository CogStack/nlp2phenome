FROM python:3.6

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

RUN mkdir /app
RUN cd /app
WORKDIR /app

CMD [ "python", "./nlp_to_phenome.py" ]