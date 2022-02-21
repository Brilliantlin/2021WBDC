
#src1
python ./src1/prepare/prepare_data.py
python ./src1/train/train_deepfm.py
python ./src1/train/train_lgb.py

# src2
python ./src2/prepare/LGB-feat-B.py
python ./src2/train/lgb-base.py

# src3
python ./src3/prepare/jin_feat.py
python ./src3/train/lgb_train.py