from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN1"))
api.upload_folder(
    folder_path="cp_predictive_maintenance_proj/deployment",     # the local folder containing your files
    repo_id="ravikmrg6/CapStnProjMlopsPred",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
