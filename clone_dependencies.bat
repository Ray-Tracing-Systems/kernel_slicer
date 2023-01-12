#!/usr/bin/bash
cd apps
git clone --depth 1 https://github.com/zeux/volk.git
git clone --depth 1 https://github.com/msu-graphics-group/vk-utils.git vkutils
cd volk && cmake . && make && cd ../..

