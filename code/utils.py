import torch
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import polarTransform
from scipy.ndimage import zoom

def sample_and_infer(generator, dataset, device):
    generator.eval()  # Set model to evaluation mode

    # Sample two random indices
    idx1, idx2 = torch.randint(0, len(dataset), (2,)).tolist()

    # Get the corresponding data
    mean_a_scan1, target_b_scan1 = dataset[idx1]
    mean_a_scan2, target_b_scan2 = dataset[idx2]

    # Add batch dimension and move to device
    mean_a_scan1 = mean_a_scan1.unsqueeze(0).to(device)
    mean_a_scan2 = mean_a_scan2.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        generated_b_scan1 = generator(mean_a_scan1).squeeze(0).cpu()
        generated_b_scan2 = generator(mean_a_scan2).squeeze(0).cpu()

    # Convert tensors to NumPy for visualization
    mean_a_scan_np1 = mean_a_scan1.squeeze(0).cpu().numpy()
    target_b_scan_np1 = target_b_scan1.cpu().numpy()
    generated_b_scan_np1 = generated_b_scan1.numpy()

    mean_a_scan_np2 = mean_a_scan2.squeeze(0).cpu().numpy()
    target_b_scan_np2 = target_b_scan2.cpu().numpy()
    generated_b_scan_np2 = generated_b_scan2.numpy()

    # Plot the results for both samples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # First Example (Row 1)
    axes[0, 0].imshow(mean_a_scan_np1[0], cmap='gray')
    axes[0, 0].set_title("Input Mean A-Scan 1")
    axes[0, 1].imshow(target_b_scan_np1[0], cmap='gray')
    axes[0, 1].set_title("Ground Truth B-Scan 1")
    axes[0, 2].imshow(generated_b_scan_np1[0], cmap='gray')
    axes[0, 2].set_title("Generated B-Scan 1")

    # Second Example (Row 2)
    axes[1, 0].imshow(mean_a_scan_np2[0], cmap='gray')
    axes[1, 0].set_title("Input Mean A-Scan 2")
    axes[1, 1].imshow(target_b_scan_np2[0], cmap='gray')
    axes[1, 1].set_title("Ground Truth B-Scan 2")
    axes[1, 2].imshow(generated_b_scan_np2[0], cmap='gray')
    axes[1, 2].set_title("Generated B-Scan 2")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def compute_metrics(model_path, val_loader, lpips, len_set,ssim ,DEVICE):
    gen = torch.load(model_path, weights_only=False).to(DEVICE)
    gen.eval()
    with torch.no_grad():
        test_ssim = 0.0
        test_lpips = 0.0
        for a_scan, real_b_scan in tqdm(val_loader, desc="Computing SSIM and LPIPS"):
            fake_b_scan = gen(a_scan.to(DEVICE))
            ssim_value = ssim(fake_b_scan.to(DEVICE), real_b_scan.to(DEVICE))
            lpips_value = lpips(fake_b_scan.to(DEVICE), real_b_scan.to(DEVICE))
            test_ssim += ssim_value
            test_lpips+=lpips_value
        test_ssim /= len(len_set)
        test_lpips /= len(len_set)
        print(f'SSIM Value is: {test_ssim.item():.3f}')
        print(f'LPIPS Value is: {test_lpips.item():.3f}')

