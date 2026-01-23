# Connect via SSH
ssh ...

# clone Repo
git clone https://github.com/lifelonglearner94/masterthesis_training_pipeline.git

# Move to folder
cd masterthesis_training_pipeline/

# installing uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# installing gdown
pip install gdown

# download .env and data
gdown 1qRb4HJjFQDNwyFqjUiTzJucvCbZpreJY

cd data/ && gdown 1yOOUrCsAfuDxLP58Wp8oRuwQNbNz6htu

# Unzip

unzip clips_0_to_5000.zip

# install dependencies
uv sync

# run training
cd ..
uv run src/train.py experiment=vjepa2_ac data/clips_0_to_5000


---

# To fetch the data to my local filesystem (execute in local terminal):
scp -P 48280 -r root@212.85.84.41:/workspace/masterthesis_training_pipeline/outputs .
