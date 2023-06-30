echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --statistics
echo python prepare-data.py --train-data



echo python train.py -e 2
python train.py -e 12
python train.py -e 22

python predict.py -e 2
python predict.py -e 12
python predict.py -e 22

python evaluate.py -e 2
python evaluate.py -e 12
python evaluate.py -e 22