sudo apt-add-repository non-free
sudo apt update
sudo apt install unrar

wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar

wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip