#!/bin/zsh -i
# Created: Jun, 22, 2026 16:35:20 by Wataru Fukuda
set -eu

BASE=$(dirname "$0")

$BASE/2D_shape_function_visu.py
lualatex $BASE/compare_shape_functions.tex
lualatex $BASE/compare_shape_functions.tex
rm *.aux *.log
open compare_shape_functions.pdf

