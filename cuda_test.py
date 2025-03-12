import torch

# Informazioni di base su Torch e CUDA
print("Torch path:", torch.__file__)
print("Cuda available?", torch.cuda.is_available())
print("Devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    # Stampa del dispositivo GPU attualmente in uso
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    print("Dispositivo GPU individuato:", gpu_name)
    
    # Test per verificare che un'operazione venga eseguita sulla GPU
    # Creiamo un tensore e lo spostiamo su GPU
    tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
    tensor_gpu = tensor_cpu.to("cuda")
    
    # Eseguiamo un'operazione (moltiplicazione per 2) sul tensore GPU
    result_gpu = tensor_gpu * 2.0
    
    # Verifichiamo che il risultato si trovi effettivamente sulla GPU
    if result_gpu.device.type == 'cuda':
        print("Test superato: l'operazione è stata eseguita sulla GPU.")
    else:
        print("Test fallito: l'operazione non è stata eseguita sulla GPU.")
else:
    print("CUDA non è disponibile. Test non eseguito.")
