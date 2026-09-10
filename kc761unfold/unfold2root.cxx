// unfold2root.cxx
// Convert a kc761unfold export file (written by kc761unfold.export) into a
// ROOT file:
//   1. kc761_spectrum_unfolded  TH1D   unfolded mu (or raw counts in
//         calibration-only mode) on the variable-width energy axis;
//         fSumw2 = per-bin total uncertainty.
//   2. sigma_statistical        TH1D   per-bin statistical 1-sigma
//   3. sigma_systematic         TH1D   per-bin systematic 1-sigma
//         (calibration parameters + simulation-file Monte Carlo)
//         The bands carry every bin, including the zero-content/bound
//         bins (their errors are 0).
//   4. kc761_spectrum_refolded  TH1D   refolded spectrum R mu
//   5. chi2 / pen_cost          TParameter<double>;
//      ndof / n_iter            TParameter<int>    (unfold mode only)
//   6. settings:
//      alpha / elo / ehi / mask_z0 / mask_floor / syst_frac
//        TParameter<double>   (mask_z0 / mask_floor / syst_frac in unfold
//        mode only)
//      chlo / chhi / calib_only / k / snip_iter  TParameter<int>
//        (k / snip_iter in unfold mode only)
// Usage:  root -l -b -q 'unfold2root.cxx("export.kc761unfold","out.root")'

#include "../kc761util/rootmacros.h"
#include "TH1D.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char kMagic[] = "kc761unfold export v2\n";

} // namespace

void unfold2root(const std::string& exportFile, const std::string& output) {
    std::FILE* f = std::fopen(exportFile.c_str(), "rb");
    if (!f) {
        std::cerr << "[unfold2root] error: cannot open export file: "
                  << exportFile << "\n";
        gSystem->Exit(1);
        return;
    }
    const auto bail = [&](const std::string& message) {
        std::cerr << "[unfold2root] error: " << message << "\n";
        std::fclose(f);
        gSystem->Exit(1);
    };

    std::string magic;
    if (!ReadLine(f, magic) || magic != kMagic)
        bail("not a kc761unfold export (bad magic line)");

    auto readI64 = [&](const char* what) -> int64_t {
        int64_t v = 0;
        if (!ReadRaw(f, &v, sizeof(v))) bail(what);
        return v;
    };

    const int64_t mode = readI64("malformed mode block");
    const int64_t nBins = readI64("malformed n_bins block");
    const int64_t chlo = readI64("malformed channel_low block");
    const int64_t chhi = readI64("malformed channel_high block");
    if (mode != 0 && mode != 1) bail("bad mode value");
    if (nBins <= 0 || nBins > (1 << 14))
        bail("implausible n_bins");
    if (chlo < 0 || chhi < chlo) bail("bad channel range");

    std::vector<double> edges(static_cast<size_t>(nBins) + 1);
    std::vector<double> counts(static_cast<size_t>(nBins));
    std::vector<double> sigmaTotal(static_cast<size_t>(nBins));
    std::vector<double> sigmaStat(static_cast<size_t>(nBins));
    std::vector<double> sigmaSyst(static_cast<size_t>(nBins));
    if (!ReadRaw(f, edges.data(), edges.size() * sizeof(double)) ||
        !ReadRaw(f, counts.data(), counts.size() * sizeof(double)) ||
        !ReadRaw(f, sigmaTotal.data(), sigmaTotal.size() * sizeof(double)) ||
        !ReadRaw(f, sigmaStat.data(), sigmaStat.size() * sizeof(double)) ||
        !ReadRaw(f, sigmaSyst.data(), sigmaSyst.size() * sizeof(double)))
        bail("malformed spectrum block");
    for (int64_t i = 0; i < nBins; ++i) {
        if (!std::isfinite(counts[i]) || !std::isfinite(sigmaTotal[i]) ||
            !std::isfinite(sigmaStat[i]) || !std::isfinite(sigmaSyst[i]))
            bail("non-finite spectrum entry");
    }

    const int64_t hasRefolded = readI64("malformed refolded flag");
    std::vector<double> refolded;
    if (hasRefolded) {
        refolded.assign(static_cast<size_t>(nBins), 0.0);
        if (!ReadRaw(f, refolded.data(), refolded.size() * sizeof(double)))
            bail("malformed refolded block");
    }

    double scalars[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // alpha, mask_z0, mask_floor, syst_frac, elo, ehi
    if (!ReadRaw(f, scalars, sizeof(scalars))) bail("malformed settings block");
    const int64_t k = readI64("malformed k block");
    const int64_t snipIter = readI64("malformed snip_iter block");
    double stats[2] = {0.0, 0.0}; // chi2, pen_cost
    if (!ReadRaw(f, stats, sizeof(stats))) bail("malformed statistics block");
    const int64_t ndof = readI64("malformed ndof block");
    const int64_t nIter = readI64("malformed n_iter block");

    TFile fOut(output.c_str(), "RECREATE");
    if (fOut.IsZombie()) {
        std::cerr << "[unfold2root] error: cannot create output file: "
                  << output << "\n";
        std::fclose(f);
        gSystem->Exit(1);
        return;
    }
    fOut.cd();

    std::vector<double> edgesCopy(edges); // TH1D takes a non-const array
    const char* specTitle = (mode == 0) ? "unfolded spectrum" :
                                          "calibrated spectrum";
    TH1D* hSpec = new TH1D("kc761_spectrum_unfolded",
                           (std::string(specTitle) +
                            ";Energy (keV);Counts")
                               .c_str(),
                           static_cast<int>(nBins), edgesCopy.data());
    hSpec->Sumw2();
    for (int64_t i = 0; i < nBins; ++i) {
        hSpec->SetBinContent(static_cast<int>(i + 1), counts[i]);
        hSpec->SetBinError(static_cast<int>(i + 1), sigmaTotal[i]);
    }
    hSpec->Write();

    if (hasRefolded) {
        TH1D* hRef = new TH1D("kc761_spectrum_refolded",
                              "refolded spectrum;Energy (keV);Counts",
                              static_cast<int>(nBins), edgesCopy.data());
        for (int64_t i = 0; i < nBins; ++i)
            hRef->SetBinContent(static_cast<int>(i + 1), refolded[i]);
        hRef->Write();
    }

    if (mode == 0) {
        // Per-bin uncertainty bands; every bin is recorded, including
        // the zero-content/bound bins (their errors are 0).
        TH1D* hStat = new TH1D("sigma_statistical",
                               "statistical uncertainty;Energy (keV);"
                               "1-sigma",
                               static_cast<int>(nBins), edgesCopy.data());
        TH1D* hSyst = new TH1D("sigma_systematic",
                               "systematic uncertainty;Energy (keV);1-sigma",
                               static_cast<int>(nBins), edgesCopy.data());
        for (int64_t i = 0; i < nBins; ++i) {
            hStat->SetBinContent(static_cast<int>(i + 1), sigmaStat[i]);
            hSyst->SetBinContent(static_cast<int>(i + 1), sigmaSyst[i]);
        }
        hStat->Write();
        hSyst->Write();
    }

    TParameter<double>("alpha", scalars[0]).Write();
    TParameter<double>("elo", scalars[4]).Write();
    TParameter<double>("ehi", scalars[5]).Write();
    TParameter<int>("chlo", static_cast<int>(chlo)).Write();
    TParameter<int>("chhi", static_cast<int>(chhi)).Write();
    TParameter<int>("calib_only", static_cast<int>(mode)).Write();
    if (mode == 0) {
        TParameter<double>("mask_z0", scalars[1]).Write();
        TParameter<double>("mask_floor", scalars[2]).Write();
        TParameter<double>("syst_frac", scalars[3]).Write();
        TParameter<int>("k", static_cast<int>(k)).Write();
        TParameter<int>("snip_iter", static_cast<int>(snipIter)).Write();
        TParameter<double>("chi2", stats[0]).Write();
        TParameter<double>("pen_cost", stats[1]).Write();
        TParameter<int>("ndof", static_cast<int>(ndof)).Write();
        TParameter<int>("n_iter", static_cast<int>(nIter)).Write();
    }

    fOut.Close();
    std::fclose(f);
    std::remove(exportFile.c_str());
    std::cout << "[unfold2root] wrote " << output << " : mode " << mode
              << ", " << nBins << " energy bins, channels " << chlo << "-"
              << chhi << "\n";
}
