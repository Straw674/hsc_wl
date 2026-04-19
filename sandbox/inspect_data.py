# %%
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        print(f"Project root found: {root_path}")  # Confirm the path found
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)

    if root_path_str not in sys.path:
        sys.path.append(root_path_str)

    from initial import *  # noqa: F403

else:
    print("Could not proceed without finding the project root.")

# %%

data = Table.read(root_path / "data/nz.fits")
print(data[0])
print(len(data))


fig, axs = plt.subplots(2, 2, figsize=(10, 6))
for i in range(4):
    axs.flatten()[i].plot(data["Z_MID"], data[f"BIN{i + 1}"])
    axs.flatten()[i].set_title("BIN" + str(i + 1))
    axs.flatten()[i].set_xlim(0, 2)
plt.show()

# %%

shape = Table.read(root_path / "data/hscy3_cat.fits")
print(shape.colnames)
print(len(shape))

# %%

hecto = Table.read(root_path / "data/HECTOMAP.fits")
print(hecto.colnames)

# unique and numbers of hecto['hsc_y3_zbin']
print(np.unique(hecto["hsc_y3_zbin"]))
for zbin in np.unique(hecto["hsc_y3_zbin"]):
    print(f"zbin {zbin}: {np.sum(hecto['hsc_y3_zbin'] == zbin)}")

# %%

random = Table.read(root_path / "data/random_hectomap.fits")
print(random.colnames)
random.remove_columns([col for col in random.colnames if col.endswith("_isnull")])
print(random.colnames)

# save to the original file
# random.write(root_path / "data/random_hectomap.fits", overwrite=True)
