#!/bin/bash
for i in {1..10}
do
    echo "=========== Log Data ==========="
    echo "Now iteration is"
    echo $i
    python -u demo.py $i
done