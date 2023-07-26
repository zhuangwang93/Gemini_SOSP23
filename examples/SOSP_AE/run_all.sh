#!/bin/bash
cd -

# kill training manually
# python3 ~/zhuang/Gemini/deepspeed/runtime/snapshot/launch.py -m kill

models=('GPT' 'BERT' 'RobertaLM')
for model in "${models[@]}"
do 
    echo $model
    cd ../$model
    bash launch.sh
    cd -
    sleep 5
done

python3 extract_results.py