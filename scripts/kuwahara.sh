#!/bin/bash

i=0
for filename in ./pawel-resized/*.jpg; do
    convert "$filename" -kuwahara 3\> ./pawel-kuwahara/pawel$i.jpg
    ((i++))
done

j=0
for filename in ./madonna-resized/*.jpg; do
    convert "$filename" -kuwahara 3\> ./madonna-kuwahara/madonna$j.jpg
    ((j++))
done

k=0
for filename in ./obama-resized/*.jpg; do
    convert "$filename" -kuwahara 3\> ./obama-kuwahara/obama$k.jpg
    ((k++))
done
