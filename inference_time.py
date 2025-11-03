import torch
import time
import numpy as np

def inference_time(model, test_loader, device):
    ''' This function is meant to calculate the inference time, the time per sample
    and samples per second of the model'''
    inf_time = []
    samples = 0

    with torch.no_grad():
        start_time = time.time()
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            samples += len(images)

        inf_time.append(time.time() - start_time)

    print(f'Samples: {samples}')

    avg_time = np.mean(inf_time)
    predictions_per_second = samples / avg_time
    time_per_prediction = avg_time / samples * 1000 # ms

    print('Inference time')
    print('\n')
    print(f'\nTime per sample: {time_per_prediction: .4f} ms')
    print(f'\nSamples per second: {predictions_per_second: .2f}')
