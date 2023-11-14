import torch;

print("torch.version installed",torch.__version__);
print("cuda version required",torch.version.cuda)
print('cuda available:', torch.cuda.is_available());
if(torch.cuda.is_available()):
    print('cuda device name:', torch.cuda.get_device_name(torch.device('cuda')));
