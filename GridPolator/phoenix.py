def read_phoenix(
    teff: u.Quantity,
    R: int,
    w1: u.Quantity,
    w2: u.Quantity
) -> Tuple[u.Quantity, u.Quantity]:
    """
    Read a PHOENIX model and return an appropriately binned version

    Parameters
    ----------
    teff : astropy.units.Quantity
        The effective temperature of the model.
    R : int
        The desired resolving power.
    w1 : astropy.units.Quantity
        The blue wavelength limit.
    w2 : astropy.units.Quantity
        The red wavelenght limit.

    Returns
    -------
    wl_new : astropy.units.Quantity
            The wavelength axis of the model.
    fl_new : astropy.units.Quantity
        The flux values of the model.
    """
    binned_options = get_binned_options()
    options_gte = binned_options >= R
    if not np.any(options_gte):
        wl, flux = RawReader().read(teff)
    else:
        binned_R = np.min(binned_options[options_gte])
        wl, flux = BinnedReader().read(binned_R, teff)
    wl_new: u.Quantity = get_wavelengths(R, w1.to_value(
        config.wl_unit), w2.to_value(config.wl_unit))*config.wl_unit
    try:
        fl_new = bin_spectra(
            wl_old=wl.to_value(config.wl_unit),
            fl_old=flux.to_value(config.flux_unit),
            wl_new=wl_new.to_value(config.wl_unit)
        )*config.flux_unit
    except ValueError:
        interp = RegularGridInterpolator(
            points=[wl.to_value(config.wl_unit)],
            values=flux.to_value(config.flux_unit)
        )
        fl_new = interp(wl_new.to_value(config.wl_unit))*config.flux_unit
    return wl_new, fl_new