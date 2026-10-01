# Reference cluster catalogs

`reference_catalogs/raw/` contains the public release files. The corresponding `*_y3.parquet` tables retain native catalog centers inside `mask/hsc_y3_mask_nside8192.hs`, the boolean HSC Y3 shape mask at NSIDE=8192. The comparison HTML displays objects with 0.19 ≤ z ≤ 0.52. Download URLs, selections, counts, and the mask checksum are recorded in `reference_catalogs/manifest.json`.

| Catalog | Release | Selection and coordinates | Source |
| --- | --- | --- | --- |
| ACT SZ | DR6 v1.0 | Full optically confirmed catalog; SZ centers; adopted redshift; flags retained | [NASA LAMBDA](https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_szcluster_catalog_info.html) |
| eROSITA eRASS1 | Primary v3.2 + Kluge et al. (2024) optical table | X-ray centers and BEST_Z; eROMaPPer optical properties joined by DETUID for the same clusters | [eROSITA DR1](https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/) |
| eROSITA eFEDS | Liu v3.2 + Klein v2.1 | X-ray centers; catalog redshift; 0 ≤ F_CONT_BEST_COMB < 0.3 | [eROSITA EDR](https://erosita.mpe.mpg.de/edr/eROSITAObservations/Catalogues/) |
| KiDS AMICO | DR3, CDS J/A+A/665/A100 (Lesci et al. 2022) | Public subset with intrinsic richness ≥ 15 and S/N ≥ 3.5; AMICO centers; corrected zfix | [CDS](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A%2BA/665/A100) |
| DES redMaPPer | Y3 v6.4.22+2 | Official index/redmapper/lgt20/select; optical centers; z_lambda | [DES Y3 release](https://data.darkenergysurvey.org/fnalmisc/y3-clusters/) |
| DES WaZP | Y6 v5.0.12.6801 | IN_COSMO=True and NGALS ≥ 25; density-map centers; ZPHOT | [LIneA release](https://data.linea.org.br/en/sci_products/wazp.html) |
| XXL | DR2, updated CDS IX/52/xxl365gc | Confirmed C1/C2 clusters; X-ray centers; catalog redshift | [CDS](https://cdsarc.cds.unistra.fr/viz-bin/cat/IX/52) |

Prepare the masked references with `uv run python scripts/data_process/prepare_reference_catalogs.py`. The reference layers are used for spatial comparison; the matching and rank-tier statistics use the prepared main lens catalogs.
