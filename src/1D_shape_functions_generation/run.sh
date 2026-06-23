#!/bin/zsh -i
# Created: Jun, 23, 2026 15:52:14 by Wataru Fukuda
set -eu

BASE=$(dirname "$0")

$BASE/1D_shape_functions_generation.py
lualatex $BASE/1D_shape_functions.tex
lualatex $BASE/1D_shape_functions.tex
rm *.aux *.log
open 1D_shape_functions.pdf

