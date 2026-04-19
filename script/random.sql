SELECT
    r.object_id,
    r.ra,
    r.dec
FROM
    pdr3_wide.random as r
    LEFT JOIN pdr3_wide.random_masks as rm USING (object_id)
WHERE
    boxSearch(r.coord, 200, 250, 42, 44.5)
    AND r.adjust_density < 0.0001
    AND r.isprimary = True
    AND NOT r.g_pixelflags_edge
    AND NOT r.g_pixelflags_interpolatedcenter
    AND NOT r.g_pixelflags_saturatedcenter
    AND NOT r.r_pixelflags_edge
    AND NOT r.r_pixelflags_interpolatedcenter
    AND NOT r.r_pixelflags_saturatedcenter
    AND NOT r.i_pixelflags_edge
    AND NOT r.i_pixelflags_interpolatedcenter
    AND NOT r.i_pixelflags_saturatedcenter
    AND NOT r.z_pixelflags_edge
    AND NOT r.z_pixelflags_interpolatedcenter
    AND NOT r.z_pixelflags_saturatedcenter
    AND NOT r.y_pixelflags_edge
    AND NOT r.y_pixelflags_interpolatedcenter
    AND NOT r.y_pixelflags_saturatedcenter
    AND NOT rm.g_mask_brightstar_halo
    AND NOT rm.g_mask_brightstar_ghost
    AND NOT rm.g_mask_brightstar_blooming
    AND NOT rm.r_mask_brightstar_halo
    AND NOT rm.r_mask_brightstar_ghost
    AND NOT rm.r_mask_brightstar_blooming
    AND NOT rm.i_mask_brightstar_halo
    AND NOT rm.i_mask_brightstar_ghost
    AND NOT rm.i_mask_brightstar_blooming
    AND NOT rm.z_mask_brightstar_halo
    AND NOT rm.z_mask_brightstar_ghost
    AND NOT rm.z_mask_brightstar_blooming
    AND NOT rm.y_mask_brightstar_halo
    AND NOT rm.y_mask_brightstar_ghost
    AND NOT rm.y_mask_brightstar_blooming;