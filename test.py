import h5py
import os

dataset_dir = "/mnt/inspurfs/evla1_t/lijiarui/datasets/dual_arm_pickings_collector0_20251101"
corrupted = []
valid = []

for f in sorted(os.listdir(dataset_dir)):
    if f.endswith('.hdf5'):
        path = os.path.join(dataset_dir, f)
        try:
            with h5py.File(path, 'r') as hf:
                _ = list(hf.keys())
            valid.append(f)
            print(f"✓ {f}")
        except Exception as e:
            corrupted.append(f)
            print(f"✗ {f} - {e}")

print(f"\n有效文件: {len(valid)}")
print(f"损坏文件: {len(corrupted)}")
if corrupted:
    print("\n需要删除:")
    for f in corrupted:
        print(f"  rm {os.path.join(dataset_dir, f)}")