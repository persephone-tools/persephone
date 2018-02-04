FROM mtgupf/essentia:ubuntu16.04-python3
#FROM python:3.6 What's the reason for the mtgupf/essentia base image?

RUN apt-get update && apt-get -y install \
	python3-pip \
	ffmpeg \
	wget \
	unzip \
	vim

RUN pip3 install -U pip
RUN pip3 install git+git://github.com/oadams/persephone.git

WORKDIR /persephone

ADD https://www.dropbox.com/s/d0vvgv0b762ck9q/na_example.zip?dl=1 data/

RUN unzip data/na_example.zip -d data/ && rm data/na_example.zip
