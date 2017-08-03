#!/bin/sh 
nvcc -std=c++11 calc.cu -o calc -res-usage -O2 -Xcompiler "-fopenmp"
clang++ main.cpp -o main -std=c++1z -lstdc++fs -O2 -fopenmp
