#!/usr/bin/bash
git submodule init 
git submodule update
cd apps/vkutils && git checkout slicer && cd ../..
cd apps/volk && cmake . && make && cd ../..

