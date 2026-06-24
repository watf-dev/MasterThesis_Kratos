#!/bin/sh
# Created: Jun, 24, 2026 14:29:41 by Wataru Fukuda
set -eu

FILE=$1
STEM=$(basename ${FILE%.*})

lualatex $FILE
pdfcrop --margin 5 $STEM.pdf $STEM.pdf
mv $STEM.pdf figs/
open figs/$STEM.pdf
rm *.aux *.log
