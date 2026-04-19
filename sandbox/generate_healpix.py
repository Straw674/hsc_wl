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

    try:
        from initial import *
    except ModuleNotFoundError as e:
        print(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")

# %%
random = Table.read(root_path / "data/random_hectomap.fits")


# %%
def choose_nside_by_target_occupancy(
    ra, dec, target_per_pixel=8.0, min_nside=32, max_nside=4096
):
    """
    Pick an nside from powers of two based on median source occupancy per occupied pixel.

    Criterion:
    - Compute occupied pixels for each candidate nside.
    - Choose nside with median(count per occupied pixel) closest to target_per_pixel.
    """
    if len(ra) == 0:
        raise ValueError("Empty RA/Dec input.")

    min_order = int(np.log2(min_nside))
    max_order = int(np.log2(max_nside))
    if 2**min_order != min_nside or 2**max_order != max_nside:
        raise ValueError("min_nside and max_nside must be powers of two.")

    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))

    best_nside = None
    best_score = np.inf
    diagnostics = []

    for order in range(min_order, max_order + 1):
        nside = 2**order
        ipix = hp.ang2pix(nside, theta, phi, nest=True)
        _, counts = np.unique(ipix, return_counts=True)
        median_occ = np.median(counts)
        # Log-distance to target gives scale-invariant selection.
        score = np.abs(np.log(median_occ / target_per_pixel))
        diagnostics.append((nside, float(median_occ), float(score), int(len(counts))))
        if score < best_score:
            best_score = score
            best_nside = nside

    return best_nside, diagnostics


def make_healpix_mask(ra, dec, nside, nest=True):
    """Create a boolean HEALPix footprint mask from RA/Dec points."""
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    ipix = hp.ang2pix(nside, theta, phi, nest=nest)
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[np.unique(ipix)] = True
    return mask


def _choose_coverage_nside(nside_sparse, max_coverage=64):
    """Pick a valid HealSparse coverage nside from sparse nside."""
    coverage = 1
    while coverage * 2 <= min(max_coverage, nside_sparse):
        coverage *= 2
    if coverage == nside_sparse:
        coverage = max(1, coverage // 2)
    return coverage


def make_healsparse_mask(ra, dec, nside_sparse, nside_coverage=None, nest=True):
    """Create a HealSparseMap float mask footprint from RA/Dec points."""
    if nside_coverage is None:
        nside_coverage = _choose_coverage_nside(nside_sparse)

    if nside_coverage >= nside_sparse:
        raise ValueError("nside_coverage must be smaller than nside_sparse.")

    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    ipix = hp.ang2pix(nside_sparse, theta, phi, nest=nest)
    valid_pix = np.unique(ipix)

    hsp_map = hsp.HealSparseMap.make_empty(
        nside_coverage=nside_coverage,
        nside_sparse=nside_sparse,
        dtype=np.float32,
        sentinel=hp.UNSEEN,
    )
    hsp_map.update_values_pix(valid_pix, np.ones(valid_pix.size, dtype=np.float32))
    return hsp_map


def plot_healpix_mask_wcs(
    mask,
    nside,
    nest=True,
    nx=1200,
    ny=600,
    cmap="coolwarm",
    projection="XPH",
    center_ra=180.0,
    center_dec=0.0,
):
    """Render a HEALPix mask with an astropy.wcs celestial projection.

    Notes
    -----
    In FITS-WCS, the projection type is encoded in CTYPE, e.g.
    - RA---XPH / DEC--XPH
    - RA---HPX / DEC--HPX
    - RA---MOL / DEC--MOL
    """
    projection = str(projection).upper()
    if len(projection) != 3:
        raise ValueError("projection must be a 3-letter WCS projection code, e.g. XPH.")

    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    # Cover the full sky in the output canvas.
    w.wcs.crval = [center_ra, center_dec]
    w.wcs.cdelt = [-360.0 / nx, 180.0 / ny]

    yy, xx = np.indices((ny, nx), dtype=float)
    ra, dec = w.all_pix2world(xx, yy, 0)

    img = np.full((ny, nx), np.nan, dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    finite &= (dec >= -90.0) & (dec <= 90.0)

    theta = np.radians(90.0 - dec[finite])
    phi = np.radians(np.mod(ra[finite], 360.0))
    ipix = hp.ang2pix(nside, theta, phi, nest=nest)
    img[finite] = mask[ipix].astype(float)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection=w)
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.coords.grid(color="white", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(
        f"HEALPix Footprint in {projection} (nside={nside}, nest={nest}, "
        f"center=({center_ra:.1f}, {center_dec:.1f}))"
    )
    # cbar = plt.colorbar(im, ax=ax, pad=0.02)
    # cbar.set_label("Mask Value")
    plt.tight_layout()
    return fig, ax


# %%

# Build HEALPix + HealSparse masks with auto nside selection
best_nside, diag = choose_nside_by_target_occupancy(
    random["ra"],
    random["dec"],
    target_per_pixel=8.0,
    min_nside=32,
    max_nside=4096,
)
print(f"Auto-selected nside: {best_nside}")

hp_mask = make_healpix_mask(random["ra"], random["dec"], nside=best_nside)
hsp_mask = make_healsparse_mask(random["ra"], random["dec"], nside_sparse=best_nside)

output_dir = root_path / "output"
output_dir.mkdir(parents=True, exist_ok=True)
hsp_out = output_dir / f"healsparse_mask_nside{best_nside}.fits"
hsp_mask.write(hsp_out, clobber=True)

print(f"HEALPix occupied pixels: {np.count_nonzero(hp_mask)}")
print(f"HealSparse valid pixels: {hsp_mask.valid_pixels.size}")
print(f"Saved HealSparse mask to: {hsp_out}")

# %%
plt.figure(figsize=(12, 7))
hp.gnomview(
    hp_mask,
    rot=(230, 44, 0),
    xsize=1400,
    ysize=300,
    nest=True,
    title=f"HEALPix Mask (nside={best_nside})",
    cmap="cividis",
)
# reverse x-axis
plt.gca().invert_xaxis()
plt.show()
# %%
plot_healpix_mask_wcs(
    hsp_mask, nside=best_nside, nest=True, projection="HPX", cmap="cividis"
)
plt.gca().invert_xaxis()
plt.show()
