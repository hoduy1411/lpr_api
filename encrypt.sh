# create new key (copy keys to encrypt.py and decrypt.py)
python app/src/func/create_key.py

# encrypt model
python app/src/func/encrypt.py "app/model/lp_detector/best_lp_15082024_416_lr0.1.pt"
python app/src/func/encrypt.py "app/model/lp_recognition/lp_recognition_size256_20241211.pt"