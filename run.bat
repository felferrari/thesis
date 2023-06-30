echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --statistics
echo python prepare-data.py --train-data



python train.py -e 1
python train.py -e 3
python train.py -e 4
python train.py -e 5

python predict.py -e 1
python predict.py -e 3
python predict.py -e 4
python predict.py -e 5

python evaluate.py -e 1
python evaluate.py -e 3
python evaluate.py -e 4
python evaluate.py -e 5

python train.py -e 11
python train.py -e 13
python train.py -e 14
python train.py -e 15

python predict.py -e 11
python predict.py -e 13
python predict.py -e 14
python predict.py -e 15

python evaluate.py -e 11
python evaluate.py -e 13
python evaluate.py -e 14
python evaluate.py -e 15

python train.py -e 21
python train.py -e 23
python train.py -e 24
python train.py -e 25

python predict.py -e 21
python predict.py -e 23
python predict.py -e 24
python predict.py -e 25

python evaluate.py -e 21
python evaluate.py -e 23
python evaluate.py -e 24
python evaluate.py -e 25