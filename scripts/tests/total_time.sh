#!/usr/bin/env sh


[[ -z $1 ]] && exit 1
grep -F '/155' $1 | grep -E '^Epoch [0-9]{1,2}/50: 100' | sed 's/^.*\[//; s/<.*//' | awk -F: '{print $1*60 + $2}' | paste -sd+ - | bc
