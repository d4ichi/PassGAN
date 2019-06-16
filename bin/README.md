# Generating a RockYou sample dataset
This codes generate sample dataset in `data` directory, which was used to train the model under `pretrained` directory.


Run these in this order under this `bin` directory:
1. `get-dataset.sh`: <p>Download `rockyou-withcount.txt` and save it under `data` directory</p>

2. `make-rockyou-full.py`: <p>Duplicate password according to the counts, and save it as `rockyou-full.txt`</p>

3. `prep-data.py`: <p>Split `rockyou-full.txt` 80% / 20% for training and test dataset, and save it as `train.txt` and `test.txt`</p>
