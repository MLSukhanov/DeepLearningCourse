FROM bubthegreat/ubuntu:python-3.6.6

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y locales language-pack-ru-base

RUN sed -i 's/# ru_RU.UTF-8 UTF-8/ru_RU.UTF-8 UTF-8/' /etc/locale.gen && locale-gen ru_RU.UTF-8

ENV LANG=ru_RU.UTF-8

ENV LC_ALL=ru_RU.UTF-8

ENV PYTHONIOENCODING=utf-8

RUN pip install --upgrade pip

RUN pip install torch==1.1.0 torchvision==0.3.0

RUN pip install -r requirements.txt

COPY . .

USER root

RUN echo "Successful"

CMD ["python", "src/run_and_compare.py"]
