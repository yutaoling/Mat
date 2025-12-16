@echo off
call C:\Users\20316\anaconda3\Scripts\activate.bat mat
cd /d C:\Users\20316\PycharmProjects\Mat\LYT_OnTheFlyRL
python model_env_train_DNN.py
python model_env_train_RNN.py
pause