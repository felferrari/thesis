echo python train.py -e 1
echo python predict.py -e 1
echo python evaluate.py -e 1

echo python train.py -e 2
python predict.py -e 2
python evaluate.py -e 2

python train.py -e 3
python predict.py -e 3
python evaluate.py -e 3