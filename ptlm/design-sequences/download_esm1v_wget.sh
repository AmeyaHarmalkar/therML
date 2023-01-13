#!/bin/bash
wget \
--directory-prefix=${SCR}/torch/checkpoints \
--continue \
--input-file ./esm1v_model_urls.txt
