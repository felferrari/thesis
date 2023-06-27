echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --statistics
echo python prepare-data.py --train-data --clear-test-folder



python train.py -e 1
python train.py -e 2
echo python train.py -e 3
echo python train.py -e 4
echo python train.py -e 5

echo python predict.py -e 11

echo python evaluate.py -e 11