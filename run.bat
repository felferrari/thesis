echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --statistics
echo python prepare-data.py --train-data


python train.py -s 1 -e 12
python predict.py -s 1 -e 12
python evaluate.py -s 1 -e 12

python train.py -s 1 -e 11
python predict.py -s 1 -e 11
python evaluate.py -s 1 -e 11

