// unfold2root.cxx
// Convert a kc761unfold export file (written by kc761unfold.export) into a
// ROOT file:
//   1. kc761_spectrum_unfolded  TH1D   unfolded mu (or raw counts in
//         calibration-only mode) on the variable-width energy axis;
//         fSumw2 = per-bin total error.
//   2. stat_covariance          TH2D   statistical covariance, energy x energy
//   3. syst_covariance          TH2D   systematic (response-parameter)
//         covariance, energy x energy
//   4. kc761_spectrum_refolded  TH1D   refolded spectrum R mu
//   5. chi2 / pen_cost          TParameter<double>;
//      ndof / n_iter            TParameter<int>    (unfold mode only)
//   6. settings:
//      syst_frac / elo / ehi  TParameter<double>
//      chlo / chhi / calib_only / k / snip_iter  TParameter<int>
//      alpha / mask_z0 / mask_floor / syst_frac  TParameter<double>
//      (unfold mode only)
// Usage:  root -l -b -q 'unfold2root.cxx("export.kc761unfold","out.root")'

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TParameter.h"
#include "TSystem.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char kMagic[] = "kc761unfold export v1\n";

bool ReadLine(std::FILE* f, std::string& line) {
    line.clear();
    for (;;) {
        int c = std::fgetc(f);
        if (c == EOF) return !line.empty();
        line.push_back(static_cast<char>(c));
        if (c == '\n') return true;
    }
}

bool ReadRaw(std::FILE* f, void* buf, size_t nbytes) {
    return std::fread(buf, 1, nbytes, f) == nbytes;
}

// Fill a TH2D content array from a row-major n x n vector (row = x bin)
// and guard the TH2 linearization assumption, mirroring calib2root.cxx.
void FillMatrix(TH2D* h, const std::vector<double>& m, int64_t n,
                const char* name) {
    Double_t* arr = h->GetArray();
    const int stride = static_cast<int>(n) + 2;
    for (int64_t j = 0; j < n; ++j) {
        const int base = static_cast<int>(j + 1) * stride + 1;
        for (int64_t i = 0; i < n; ++i) {
            arr[base + static_cast<int>(i)] =
                m[static_cast<size_t>(i) * n + j];
        }
    }
    const size_t nEntries = static_cast<size_t>(n) * n;
    if (h->GetBinContent(1, 1) != m[0] ||
        h->GetBinContent(static_cast<int>(n), static_cast<int>(n)) !=
            m[nEntries - 1]) {
        std::cerr << "[unfold2root] error: ROOT bin-layout assumption "
                     "violated; "
                  << name << " would be transposed\n";
        gSystem->Exit(1);
    }
}

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
    if (!ReadRaw(f, edges.data(), edges.size() * sizeof(double)) ||
        !ReadRaw(f, counts.data(), counts.size() * sizeof(double)) ||
        !ReadRaw(f, sigmaTotal.data(), sigmaTotal.size() * sizeof(double)))
        bail("malformed spectrum block");
    for (int64_t i = 0; i < nBins; ++i) {
        if (!std::isfinite(counts[i]) || !std::isfinite(sigmaTotal[i]))
            bail("non-finite spectrum entry");
    }

    const int64_t hasRefolded = readI64("malformed refolded flag");
    std::vector<double> refolded;
    if (hasRefolded) {
        refolded.assign(static_cast<size_t>(nBins), 0.0);
        if (!ReadRaw(f, refolded.data(), refolded.size() * sizeof(double)))
            bail("malformed refolded block");
    }

    const int64_t hasStatCov = readI64("malformed stat_cov flag");
    std::vector<double> statCov;
    if (hasStatCov) {
        statCov.assign(static_cast<size_t>(nBins) * nBins, 0.0);
        if (!ReadRaw(f, statCov.data(), statCov.size() * sizeof(double)))
            bail("malformed stat_covariance block");
    }

    const int64_t hasSystCov = readI64("malformed syst_cov flag");
    std::vector<double> systCov;
    if (hasSystCov) {
        systCov.assign(static_cast<size_t>(nBins) * nBins, 0.0);
        if (!ReadRaw(f, systCov.data(), systCov.size() * sizeof(double)))
            bail("malformed syst_covariance block");
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

    if (hasStatCov) {
        TH2D* hStat = new TH2D("stat_covariance",
                               "statistical covariance;Energy (keV);Energy (keV)",
                               static_cast<int>(nBins), edgesCopy.data(),
                               static_cast<int>(nBins), edgesCopy.data());
        FillMatrix(hStat, statCov, nBins, "stat_covariance");
        hStat->Write();
    }

    if (hasSystCov) {
        TH2D* hSyst = new TH2D("syst_covariance",
                               "systematic covariance;Energy (keV);Energy (keV)",
                               static_cast<int>(nBins), edgesCopy.data(),
                               static_cast<int>(nBins), edgesCopy.data());
        FillMatrix(hSyst, systCov, nBins, "syst_covariance");
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
