"""kc761ana: analysis package for the KC761 toolkit.

Submodules:
    io          : read experimental / simulated spectra from ROOT files (uproot)
    calibrate   : energy-calibration transform E(x) = c3 x^3 + c2 x^2 + c1 x + c0
    resolution  : Gaussian resolution model sigma(E) = a2 E + a1 sqrt(E) + a0
    fitmodel    : chi^2 forward model (time-scaled sim vs calibrated data)
    fitter      : parameter fit via scipy.optimize.minimize
    plot        : PDF figures of the fit result
"""
