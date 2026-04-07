#!/bin/bash

uv run python evaluation/eval.py --FOLDER 'BEST' --no-baseline --OUTER_STEPS 1000000 

uv run python evaluation/eval.py --FOLDER 'BEST' --no-baseline --OUTER_STEPS 500000 --dim 50

uv run python evaluation/eval.py --FOLDER 'BEST' --no-baseline --OUTER_STEPS 1000000 --dim 50