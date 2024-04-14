FROM python:3.10-slim-bullseye

RUN  mkdir  /workspace &&  mkdir  /workspace/project

WORKDIR /workspace/project

COPY ./backend/requirements/neural.txt /workspace/project
ADD ./neural/deeppavlov/configs /workspace/project/neural/deeppavlov/configs

COPY ./docker_sources/en_core_web_sm-3.5.0-py3-none-any.whl /workspace/project
COPY ./docker_sources/ru_core_news_sm-3.5.0-py3-none-any.whl /workspace/project

RUN pip install h5py \
    && pip install typing-extensions \
    && pip install wheel \
    && pip install  --upgrade pybind11==2.2.4 \
    && pip install en_core_web_sm-3.5.0-py3-none-any.whl  ru_core_news_sm-3.5.0-py3-none-any.whl \
    && pip install --upgrade nltk \
    && pip install -r neural.txt

COPY ./backend/requirements/easy_neural.txt /workspace/project
COPY ./backend/requirements/install_dep.py  /workspace/project

RUN  python install_dep.py \
    && pip install --no-cache-dir  -r easy_neural.txt


#ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
