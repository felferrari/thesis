echo python tiles-gen.py
echo python previous-def-gen.py
echo python label-gen.py
echo python cloud-map-gen.py

echo python prepare-data.py --train-data --clear-test-folder
python train.py -e 21
python train.py -e 22
python train.py -e 23
python train.py -e 24
python train.py -e 25
python train.py -e 26
python train.py -e 27
python train.py -e 28
python train.py -e 29
python train.py -e 30
python train.py -e 31
python train.py -e 32
python train.py -e 33
python train.py -e 34
python train.py -e 35
python train.py -e 36
python train.py -e 37
python train.py -e 38
python train.py -e 39
python train.py -e 40

python prepare-data.py --test-data --clear-train-folder

python predict.py -e 21
python predict.py -e 22
python predict.py -e 23
python predict.py -e 24
python predict.py -e 25
python predict.py -e 26
python predict.py -e 27
python predict.py -e 28
python predict.py -e 29
python predict.py -e 30
python predict.py -e 31
python predict.py -e 32
python predict.py -e 33
python predict.py -e 34
python predict.py -e 35
python predict.py -e 36
python predict.py -e 37
python predict.py -e 38
python predict.py -e 39
python predict.py -e 40


python evaluate.py -e 21
python evaluate.py -e 22
python evaluate.py -e 23
python evaluate.py -e 24
python evaluate.py -e 25
python evaluate.py -e 26
python evaluate.py -e 27
python evaluate.py -e 28
python evaluate.py -e 29
python evaluate.py -e 30
python evaluate.py -e 31
python evaluate.py -e 32
python evaluate.py -e 33
python evaluate.py -e 34
python evaluate.py -e 35
python evaluate.py -e 36
python evaluate.py -e 37
python evaluate.py -e 38
python evaluate.py -e 39
python evaluate.py -e 40