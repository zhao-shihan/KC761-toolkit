"""kc761ana: analysis package for the KC761 toolkit.

Submodules:
    io          : read experimental / simulated spectra from ROOT files (uproot)
    calibrate   : energy calibration, parameterised by the channel positions
                  of the 60/609/1461/2614 keV lines (cubic E(x))
    resolution  : Gaussian resolution model sigma(E) = a2 E + a1 sqrt(E) + a0,
                  parameterised by the relative widths sigma/E at 60/1461/2614 keV
    fitmodel    : chi^2 forward model (time-scaled sim vs calibrated data)
    fitter      : parameter fit via scipy.optimize.minimize
    plot        : PDF figures of the fit result
"""
