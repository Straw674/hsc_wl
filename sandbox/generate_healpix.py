# %%
import sys
import logging
import matplotlib as mpl
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        logger.error("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        logger.info(f"Project root found: {root_path}")  # Confirm the path found
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
        logger.error(f"Error importing 'initial': {e}")

else:
    print("Could not proceed without finding the project root.")

# %%
random = Table.read(root_path / "data/total_random.csv")
print(len(random), "random sources loaded.")


# %%
def choose_nside_by_target_occupancy(
    ra, dec, target_per_pixel=8.0, min_nside=32, max_nside=32768
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
        print(
            f"Order {order}: nside={nside}, median_occ={median_occ:.2f}, score={score:.4f}, occupied_pixels={len(counts)}"
        )
        if score < best_score:
            best_score = score
            best_nside = nside
            print(f"  -> New best nside: {best_nside} (score={best_score:.4f})")
        else:
            break

    return best_nside, diagnostics


def compute_mask_values(ra, dec, nside, nest=True):
    """Compute binary mask values for occupied pixels and unobserved sentinel.

    Rules before the final 1-x transform:
    - Occupied pixel (count > 0) -> x = 1
    - Unobserved pixel (count <= 0) -> x = 0

    Saved value is 1 - x, so occupied pixels become 0 and unobserved pixels become 1.
    """
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    ipix = hp.ang2pix(nside, theta, phi, nest=nest)
    uniq_pix, counts = np.unique(ipix, return_counts=True)

    occupied_flag = (counts > 0).astype(np.float32)
    valid_values = (1.0 - occupied_flag).astype(np.float32)

    logger.info(
        "compute_mask_values: nside=%s, occupied_pixels=%s, binary mask applied (occupied->0, unobserved->1)",
        nside,
        len(counts),
    )

    # HEALPix uses a regular binary mask with unobserved value 1.
    sentinel_value = np.float32(1.0)

    return uniq_pix, valid_values, sentinel_value


def make_healpix_mask(nside, uniq_pix, valid_values, sentinel_value):
    """Create a HEALPix mask from precomputed pixel/value arrays."""
    npix = hp.nside2npix(nside)

    mask_values = np.full(npix, sentinel_value, dtype=np.float32)
    mask_values[uniq_pix] = valid_values
    return mask_values


def _choose_coverage_nside(nside_sparse, max_coverage=64):
    """Pick a valid HealSparse coverage nside from sparse nside."""
    coverage = 1
    while coverage * 2 <= min(max_coverage, nside_sparse):
        coverage *= 2
    if coverage == nside_sparse:
        coverage = max(1, coverage // 2)
    return coverage


def make_healsparse_mask(
    nside_sparse,
    uniq_pix,
    valid_values,
    sentinel_value=np.float32(hp.UNSEEN),
    nside_coverage=None,
):
    """Create a HealSparse mask from precomputed pixel/value arrays.

    Unobserved pixels use HealSparse/HEALPix unseen sentinel by default.
    """
    if nside_coverage is None:
        nside_coverage = _choose_coverage_nside(nside_sparse)

    if nside_coverage >= nside_sparse:
        raise ValueError("nside_coverage must be smaller than nside_sparse.")

    hsp_map = hsp.HealSparseMap.make_empty(
        nside_coverage=nside_coverage,
        nside_sparse=nside_sparse,
        dtype=np.float32,
        sentinel=sentinel_value,
    )
    hsp_map.update_values_pix(uniq_pix, valid_values)
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
    _ = ax.imshow(img, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
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
    target_per_pixel=20.0,
)
print(f"Auto-selected nside: {best_nside}")

uniq_pix, valid_values, sentinel_value = compute_mask_values(
    random["ra"], random["dec"], nside=best_nside
)

hp_mask = make_healpix_mask(best_nside, uniq_pix, valid_values, sentinel_value)
hsp_mask = make_healsparse_mask(
    best_nside,
    uniq_pix,
    valid_values,
    np.float32(hp.UNSEEN),
)

print(
    "HEALPix mask stats: "
    f"min={np.nanmin(hp_mask):.3e}, median={np.nanmedian(hp_mask):.3e}, max={np.nanmax(hp_mask):.3e}"
)
print(
    "HealSparse mask stats (valid pixels): "
    f"min={np.nanmin(hsp_mask[hsp_mask.valid_pixels]):.3e}, "
    f"median={np.nanmedian(hsp_mask[hsp_mask.valid_pixels]):.3e}, "
    f"max={np.nanmax(hsp_mask[hsp_mask.valid_pixels]):.3e}"
)

output_dir = root_path / "output"
output_dir.mkdir(parents=True, exist_ok=True)
hsp_out = output_dir / f"healsparse_mask_nside{best_nside}.fits"

hp_mask_path = output_dir / f"healpix_mask_nside{best_nside}.fits"
hp.write_map(hp_mask_path, hp_mask, nest=True, overwrite=True)
print(f"Saved HEALPix mask to: {hp_mask_path}")

hsp_mask.write(hsp_out, clobber=True)
print(f"Saved HealSparse mask to: {hsp_out}")

# %%
CAMP = "cividis"


def _resample_mask(mask, nside_out, order_in="NEST", order_out="NEST"):
    """Resample a HEALPix mask to a common nside/order for fair comparison."""
    mask = np.asarray(mask, dtype=np.float32)
    nside_in = hp.npix2nside(mask.size)
    if nside_in == nside_out and order_in == order_out:
        return mask
    return hp.ud_grade(
        mask, nside_out=nside_out, order_in=order_in, order_out=order_out
    )


def compare_three_masks(mask_triplet, nside_compare, cmap="cividis_r"):
    """Compare three masks with local cartesian views on identical scale."""
    # Local comparison around HSC footprint in Cartesian projection.
    fig2 = plt.figure(figsize=(12, 5))
    for i, (title, mask) in enumerate(mask_triplet, start=1):
        hp.cartview(
            mask,
            fig=fig2.number,
            sub=(3, 1, i),
            nest=True,
            cmap=cmap,
            min=0.0,
            max=1.0,
            lonra=[200.0, 250.0],
            latra=[41, 45],
            xsize=900,
            title=f"{title} (Cartesian zoom)",
            cbar=False,
        )

    # Use a single shared colorbar to avoid unstable placement across healpy subplots.
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=fig2.axes, location="right", fraction=0.025, pad=0.02)
    cbar.set_label("Mask Value")
    fig2.subplots_adjust(left=0.03, right=0.92, bottom=0.08, top=0.90, wspace=0.20)

    plt.show()


mask1 = hp.read_map(root_path / "data/mask/fdfc_hp_window.fits", nest=True)
mask2 = hp.read_map(root_path / "data/mask/s19a_fdfc_hp_contarea_izy-gt-5_trimmed.fits")
hp_mask_for_plot = (1.0 - hp_mask).astype(np.float32)

mask_new_cmp = _resample_mask(
    hp_mask_for_plot, nside_out=best_nside, order_in="NEST", order_out="NEST"
)
mask1_cmp = _resample_mask(
    mask1, nside_out=best_nside, order_in="NEST", order_out="NEST"
)
mask2_cmp = _resample_mask(
    mask2, nside_out=best_nside, order_in="RING", order_out="NEST"
)


compare_three_masks(
    [
        (f"New mask (nside={best_nside})", mask_new_cmp),
        ("Ref mask 1: fdfc_hp_window", mask1_cmp),
        ("Ref mask 2: s19a_contarea", mask2_cmp),
    ],
    nside_compare=best_nside,
    cmap=CAMP,
)

# %%
