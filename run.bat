echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

python prepare-data.py --train-data --clear-test-folder

python train.py -e 1
python train.py -e 2

python prepare-data.py --test-data --clear-train-folder

python predict.py -e 1
python predict.py -e 2

python evaluate2.py -e 1
python evaluate2.py -e 2