#!/bin/bash

shopt -s nullglob

EXP_DIR="../exp"

cmd=$1

i=1
for f in $EXP_DIR/*.log; do
	let i=i+1
done

cp $cmd "$EXP_DIR/$i.$cmd"
python $1 > "$EXP_DIR/$i.log" 2> "$EXP_DIR/$i.stderr"
