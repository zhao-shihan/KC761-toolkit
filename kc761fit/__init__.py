"""kc761fit: spectrum-fit package for the KC761 toolkit.

Submodules:
    io          : read experimental / simulated spectra from ROOT files (uproot)
    calibration : energy calibration, parameterized by the channel positions
                  of the 60/609/1461/2614 keV lines (cubic E(x))
    resolution  : Gaussian resolution model sigma(E) = a2 E + a1 sqrt(E) + a0,
                  parameterized by the relative widths sigma/E at 60/1461/2614 keV
    fitmodel    : chi^2 forward model (normalized sim vs calibrated data)
    fitter      : parameter fit via scipy.optimize.minimize
    plot        : PDF figures of the fit result
"""
