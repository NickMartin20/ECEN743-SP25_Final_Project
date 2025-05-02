# ECEN743-SP25_Final_Project
To download an LLM from huggingface to TAMU HPRC use the execute the following lines:

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
pip install --user huggingface_hub
>
huggingface-cli login
>
python -c "
from huggingface_hub import  snapshot_download
snapshot_download(repo_id='unsloth/Llama-3.2-1B', local_dir='llama3_model')
"

