#!/bin/bash

UV_BIN="/home/jules/.local/bin/uv"


"$UV_BIN" run evaluation/eval_XML.py --OUTER_STEPS 1000000
