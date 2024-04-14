FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN  mkdir  /workspace/project

WORKDIR /workspace/project

COPY ./backend/requirements/neural.txt /workspace/project


#COPY ./backend/requirements/server.txt /workspace/project

RUN pip install h5py \
    && pip install typing-extensions \
    && pip install wheel \
    && pip install  --upgrade pybind11==2.2.4 \
    && pip install -r neural.txt

COPY ./backend/requirements/easy_neural.txt /workspace/project
COPY ./backend/requirements/install_dep.py  /workspace/project
ADD ./neural/deeppavlov/configs /workspace/project/neural/deeppavlov/configs

COPY ./docker_sources/en_core_web_sm-3.5.0-py3-none-any.whl /workspace/project
COPY ./docker_sources/ru_core_news_sm-3.5.0-py3-none-any.whl /workspace/project

RUN pip install en_core_web_sm-3.5.0-py3-none-any.whl  ru_core_news_sm-3.5.0-py3-none-any.whl \
    && python install_dep.py \
#    && python -m spacy download en_core_web_sm \
#    && python -m spacy download ru_core_web_sm \
    && pip install --no-cache-dir  -r easy_neural.txt
#    && pip install --no-cache-dir  -r server.txt
COPY ./backend/requirements/easy_neural_2.txt /workspace/project
COPY ./backend/requirements/install_dep_2.py /workspace/project
RUN pip install -r easy_neural_2.txt  \
    && pip install --upgrade nltk \
    && python install_dep_2.py

#ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
