#!/usr/bin/bash
git clone --depth 1 https://github.com/msu-graphics-group/LiteMath.git
git clone --depth 1 https://github.com/zeux/volk.git
cd apps/volk && cmake . && make && cd ../..

