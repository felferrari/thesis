echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --statistics
echo python prepare-data.py --train-data


python train.py -e 11
python predict.py -e 11
python evaluate.py -e 11

python train.py -e 12
python predict.py -e 12
python evaluate.py -e 12

python train.py -e 13
python predict.py -e 13
python evaluate.py -e 13

python train.py -e 14
python predict.py -e 14
python evaluate.py -e 14

python train.py -e 15
python predict.py -e 15
python evaluate.py -e 15

python train.py -e 16
python predict.py -e 16
python evaluate.py -e 16

python train.py -e 17
python predict.py -e 17
python evaluate.py -e 17