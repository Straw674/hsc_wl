from hsc_wl.prepare import get_latest_cluster_catalog

CATALOG_SOURCES = {
    "pdr3_redm_hsc": {
        "label": "pdr3_redm_hsc",
        "lens_path": "/Users/xinq/redmapper_HSC/output/redmapper_run/new_run_no_mask/run/hsc_run_redmapper_v0.9.1.dev2+g030802198.d20260617_lgt05_catalog.fit",
        "random_path": "data/random_hectomap.fits",
        "redshift_range": [0.10, 0.60],
        "top_counts_factor": 0.572439,
        "columns": {
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
    },
    "s16a_redm_hsc": {
        "label": "s16a_redm_hsc",
        "lens_path": "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "redshift_range": [0.19, 0.52],
        "top_counts_factor": 1.0,
        "columns": {
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
    },
    "s16a_logm_50_100": {
        "label": "s16a_logm_50_100",
        "lens_path": "/Users/xinq/redmapper_HSC/data/reference/s16a_massive_logm_11.2.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "redshift_range": [0.19, 0.52],
        "top_counts_factor": 1.0,
        "columns": {
            "col_rank": "logm_50_100",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
    },
    "s16a_redm_hsc_topn": {
        "label": "s16a_redm_hsc_topn",
        "lens_path": "/Users/xinq/redmapper_HSC/data/reference/redmapper_s16a/redmapper_hsc_s16a_cluster_bsm.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "redshift_range": [0.19, 0.52],
        "top_counts_factor": 1.0,
        "columns": {
            "col_rank": "lambda",
            "ra": "ra",
            "dec": "dec",
            "z": "z_lambda",
        },
    },
    "s16a_logm_50_100_topn": {
        "label": "s16a_logm_50_100_topn",
        "lens_path": "/Users/xinq/redmapper_HSC/data/reference/s16a_massive_logm_11.2.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "redshift_range": [0.19, 0.52],
        "top_counts_factor": 1.0,
        "columns": {
            "col_rank": "logm_50_100",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
    },
    "forced": {
        "label": "forced",
        "lens_path": "/Users/xinq/redmapper_HSC/output/s16a_massive_logm_11.2_forced_results.fits",
        "random_path": "data/s16a_weak_lensing_hdf/s16a_weak_lensing_medium_random.fits",
        "redshift_range": [0.19, 0.52],
        "top_counts_factor": 1.0,
        "columns": {
            "col_rank": "lam",
            "ra": "ra",
            "dec": "dec",
            "z": "z_best",
        },
    },
    "camira": {
        "label": "camira",
        "lens_path": "data/camira_s23b_wide_sm_v3_filtered.dat",
        "lens_format": "pandas_dat",
        "random_path": "data/random_hectomap.fits",
        "redshift_range": [0.10, 0.80],
        "ra_range": [210, 250],
        "dec_range": [42, 44.5],
        "top_counts_factor": 1.154652,
        "columns": {
            "col_rank": "N_mem",
            "ra": "RA",
            "dec": "Dec",
            "z": "z_cl",
        },
    },
    "cosine": {
        "label": "cosine",
        "lens_path": get_latest_cluster_catalog(),
        "lens_format": "parquet",
        "random_path": "data/random_hectomap.fits",
        "redshift_range": [0.10, 0.80],
        "top_counts_factor": 1.154652,
        "columns": {
            "col_rank": "true_richness",
            "ra": "ra",
            "dec": "dec",
            "z": "z_cl",
        },
    },
    "cosine_4bin": {
        "label": "cosine_4bin",
        "lens_path": get_latest_cluster_catalog(),
        "lens_format": "parquet",
        "random_path": "data/random_hectomap.fits",
        "redshift_range": [0.10, 0.80],
        "top_counts_factor": 1.154652,
        "columns": {
            "col_rank": "true_richness",
            "ra": "ra",
            "dec": "dec",
            "z": "z_cl",
        },
    },
}

RUN_PROFILES = {
    "pdr3_redm_hsc": {
        "lens_z_bins": [0.10, 0.60],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/pdr3_redm_hsc/",
    },
    "s16a_redm_hsc": {
        "lens_z_bins": [0.19, 0.52],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/s16a_redm_hsc/",
    },
    "s16a_logm_50_100": {
        "lens_z_bins": [0.19, 0.52],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/s16a_logm_50_100/",
    },
    "s16a_redm_hsc_topn": {
        "lens_z_bins": [0.19, 0.52],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/s16a_redm_hsc_topn/",
    },
    "s16a_logm_50_100_topn": {
        "lens_z_bins": [0.19, 0.52],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/s16a_logm_50_100_topn/",
    },
    "forced": {
        "lens_z_bins": [0.19, 0.52],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/forced/",
    },
    "camira": {
        "lens_z_bins": [0.10, 0.80],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/camira/",
    },
    "cosine": {
        "lens_z_bins": [0.10, 0.80],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/cosine/",
    },
    "cosine_4bin": {
        "lens_z_bins": [0.10, 0.80],
        "save_root": "/Users/xinq/dev/repos/hsc_wl/output/cosine_4bin/",
    },
}
