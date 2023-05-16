echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --train-data --clear-test-folder

python train.py -e 41
python train.py -e 42
python train.py -e 43
python train.py -e 44
python train.py -e 45

python prepare-data.py --test-data --clear-train-folder

python predict.py -e 41
python predict.py -e 42
python predict.py -e 43
python predict.py -e 44
python predict.py -e 45

python evaluate.py -e 41
python evaluate.py -e 42
python evaluate.py -e 43
python evaluate.py -e 44
python evaluate.py -e 45