#!/bin/bash

i=0
for filename in ./madonna/*.jpg; do
    convert "$filename" -resize 125x125\> ./madonna-resized/madonna$i.jpg
    ((i++))
done

j=0
for filename in ./pawel/*.jpg; do
    convert "$filename" -resize 125x125\> ./pawel-resized/pawel$i.jpg
    ((j++))
done

k=0
for filename in ./obama/*.jpg; do
    convert "$filename" -resize 125x125\> ./obama-resized/obama$i.jpg
    ((k++))
done