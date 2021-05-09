#!/usr/bin/bash

nvcc main.cpp slime.cu -o main_cuda -lsfml-graphics -lsfml-window -lsfml-system