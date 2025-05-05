import os, random, shutil

def sample_images(src_dir, dest_dir, num_samples):
    os.makedirs(dest_dir, exist_ok=True)
    all_images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    sampled = random.sample(all_images, num_samples)
    for img in sampled:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))

# Example usage
sample_images('data/train/person', 'data/train_small/person', 2000)
sample_images('data/val/person',   'data/val_small/person',   500)
sample_images('data/test/person',  'data/test_small/person',  500)
