git clone https://github.com/j-bernardi/pessimistic-agents.git
pip install -r conda_envs/tpu_requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
source set_path.sh
export STORAGE_BUCKET=gs://free-tpu-bucket
echo "Storage dir: $STORAGE_BUCKET"
