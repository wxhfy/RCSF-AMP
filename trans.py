from transformers import AutoModel
from safetensors.torch import save_file

model = AutoModel.from_pretrained("/home/fyh0106/work/utils/esm_lora/esm2/")
save_file(model.state_dict(), "/home/fyh0106/work/utils/esm_lora/esm2/model.safetensors")