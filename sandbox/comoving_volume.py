# %%
import logging
import sys
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
def solid_angle_from_radec_bounds(ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg):
    """Compute solid angle from rectangular RA/Dec bounds.

    RA can wrap around 0 deg, e.g. [350, 20].
    """
    if not (-90.0 <= dec_min_deg <= 90.0 and -90.0 <= dec_max_deg <= 90.0):
        raise ValueError("dec must be within [-90, 90] degrees")
    if dec_min_deg >= dec_max_deg:
        raise ValueError("dec_min must be smaller than dec_max")

    ra_min = ra_min_deg % 360.0
    ra_max = ra_max_deg % 360.0
    dra_deg = (ra_max - ra_min) % 360.0
    if np.isclose(dra_deg, 0.0):
        dra_deg = 360.0

    dra = np.deg2rad(dra_deg)
    dec_min = np.deg2rad(dec_min_deg)
    dec_max = np.deg2rad(dec_max_deg)

    omega = dra * (np.sin(dec_max) - np.sin(dec_min)) * u.sr
    return omega


def comoving_volume_region(
    z_min,
    z_max,
    *,
    ra_min_deg=None,
    ra_max_deg=None,
    dec_min_deg=None,
    dec_max_deg=None,
    area_deg2=None,
    cosmo=Planck18,
):
    """Comoving volume in a redshift shell and sky region.

    Choose exactly one sky description:
    1) RA/Dec bounds: ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg
    2) Sky area in deg^2: area_deg2
    """
    if z_min < 0 or z_max < 0:
        raise ValueError("z_min and z_max must be non-negative")
    if z_min >= z_max:
        raise ValueError("z_min must be smaller than z_max")

    use_area = area_deg2 is not None
    use_bounds = all(
        v is not None for v in [ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg]
    )

    if use_area == use_bounds:
        raise ValueError("Provide either area_deg2 OR all RA/Dec bounds")

    full_sky_area_deg2 = (4.0 * np.pi * u.sr).to(u.deg**2).value

    if use_area:
        if area_deg2 <= 0:
            raise ValueError("area_deg2 must be > 0")
        if area_deg2 > full_sky_area_deg2:
            raise ValueError(
                f"area_deg2 cannot exceed full sky ({full_sky_area_deg2:.3f} deg^2)"
            )
        omega = (area_deg2 * u.deg**2).to(u.sr)
    else:
        omega = solid_angle_from_radec_bounds(
            ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg
        )

    v_shell_full_sky = (cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)).to(
        u.Mpc**3
    )
    frac_sky = (omega / (4.0 * np.pi * u.sr)).value
    v_region = v_shell_full_sky * frac_sky

    return {
        "cosmology": cosmo.name,
        "z_min": z_min,
        "z_max": z_max,
        "omega_sr": omega.to_value(u.sr),
        "area_deg2": omega.to_value(u.deg**2),
        "sky_fraction": frac_sky,
        "volume_mpc3": v_region.to_value(u.Mpc**3),
        "volume_gpc3": v_region.to_value(u.Gpc**3),
    }


result = comoving_volume_region(
    0.19,
    0.52,
    area_deg2=137,
)["volume_gpc3"]

print(result)

result2 = comoving_volume_region(0.1, 0.6, area_deg2=72.8)["volume_gpc3"]
print(result2)

print(f"Ratio of volumes: {result2 / result:.6f}")

# %%
for z in np.linspace(0.1, 0.6, 6):
    v = Planck18.comoving_volume(z).to(u.Gpc**3)
    print(f"Comoving volume to z={z}: {v:.3f}")

# %%
z_vals = np.linspace(0.01, 1.5, 100)
volumes = [Planck18.comoving_volume(z).to(u.Gpc**3).value for z in z_vals]
plt.plot(z_vals, volumes)
plt.xlabel("Redshift z")
plt.ylabel("Comoving Volume (Gpc^3)")
plt.title("Comoving Volume vs Redshift for Full Sky")
plt.grid()
plt.show()
