#!/bin/bash
aria2c \
-d ${SCR}/.cache/torch/hub/checkpoints \
--max-concurrent-downloads=8 \
--continue \
--max-connection-per-server=8 \
--file-allocation=none \
--auto-file-renaming=false \
--input-file ./esm1v_model_urls.txt
