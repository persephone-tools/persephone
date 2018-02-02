FROM mtgupf/essentia:ubuntu16.04-python3

RUN apt-get update && apt-get -y install \
	python3-pip \
    ffmpeg \
    virtualenv \
    wget \
    unzip

RUN git clone https://github.com/oadams/persephone.git /persephone

WORKDIR /persephone

ADD https://www.dropbox.com/s/d0vvgv0b762ck9q/na_example.zip?dl=1 data/

RUN pip3 install -r requirements.txt && \
	unzip data/na_example.zip && \
	rm data/na_example.zip
