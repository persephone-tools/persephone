#!/bin/bash

for file in *mp3
do
	ffmpeg -i $file -ar 16000 -ac 1 ${file%.mp3}.wav
done
