python previous-def-gen.py -s 1
python label-gen.py -s 1
python tiles-gen.py -s 1
python prepare-data.py -s 1 -r


python train.py -s 1 -e 6
python predict.py -s 1 -e 6
python evaluate.py -s 1 -e 6
