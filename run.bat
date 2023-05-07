echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python prepare-data.py
python train.py -e 1
python train.py -e 2
python train.py -e 3

python predict.py -e 1
python predict.py -e 2
python predict.py -e 3

python train.py -e 4
python train.py -e 5

python predict.py -e 4
python predict.py -e 5