#!/bin/bash

i=0
for filename in ./pawel-resized/*.jpg; do
    convert "$filename" -edge 10\> ./pawel-edge/pawel$i.jpg
    ((i++))
done

j=0
for filename in ./madonna-resized/*.jpg; do
    convert "$filename" -edge 10\> ./madonna-edge/madonna$j.jpg
    ((j++))
done

k=0
for filename in ./obama-resized/*.jpg; do
    convert "$filename" -edge 10\> ./obama-edge/obama$k.jpg
    ((k++))
done
