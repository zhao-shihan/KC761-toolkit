// calib2root.cxx
// Convert the kc761calib binary export (written by kc761calib/export.py)
// into the calibration ROOT file.  Internal to kc761calib.
//
// Objects are written in this order:
//   TNamed              "calib_formula"   calibration formula text
//   TParameter<double>  "c0".."c3"        cubic calibration coefficients,
//                       each immediately followed by its 1-sigma error
//                       "c0_err".."c3_err"
//   TNamed              "resol_formula"   resolution formula text
//   TParameter<double>  "resol_e_ref"     resolution reference energy (keV)
//   TParameter<double>  "b0".."b2"        resolution parameters (keV), each
//                       immediately followed by its 1-sigma error
//                       "b0_err".."b2_err"
//   TNamed              "param_order"     parameter order of param_cov:
//                                          "c0 c1 c2 c3 b0 b1 b2"
//   TMatrixDSym         "param_cov"       7x7 covariance of the stored
//                                          parameters in the param_order
//                                          basis (calib-resol cross block
//                                          included; NaN rows/columns mark
//                                          undetermined parameters)
//   TH2D                "deposition_response_matrix"
//                       x = detected channel (uniform bins of width 1,
//                       integer bin centers), y = energy deposition
//                       (variable-width bins from the calibration image of
//                       the channel bins); content R[ch, Edep] =
//                       probability that a count in the energy-deposition
//                       bin Edep is detected in channel bin ch.  The
//                       columns are NOT renormalized: the Gaussian
//                       probability beyond the detector channel range is
//                       truncated, i.e. physically lost.  The per-bin
//                       errors (fSumw2) hold the per-element 1-sigma
//                       uncertainty propagated linearly from param_cov.
//
// The temporary export file is deleted after a successful write; on error
// it is left in place so the caller can inspect or re-run it.
//
// Usage:  root -l -b -q 'calib2root.cxx("export.tmp","out.root")'

#include "../kc761util/rootmacros.h"

#include "TFile.h"
#include "TH2D.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kMagic = "kc761calib-export-v2\n";
const int kNC = 4;                    // calibration coefficients c0..c3
const int kNB = 3;                    // resolution parameters b0..b2
const int kNCore = 7;                 // stored parameters (c0..c3, b0..b2)
const int64_t kMaxChannels = 1 << 14; // Sanity cap for the channel count. 8 * 16384^2 = 2 GiB per block

} // namespace

void calib2root(const std::string& exportFile, const std::string& output) {
    std::FILE* f = std::fopen(exportFile.c_str(), "rb");
    if (!f) {
        std::cerr << "[calib2root] error: cannot open export file: "
                  << exportFile << "\n";
        gSystem->Exit(1);
        return;
    }
    // Leave the export file in place on error; delete it only after a
    // successful write.
    const auto bail = [&](const std::string& message) {
        std::cerr << "[calib2root] error: " << message << "\n";
        std::fclose(f);
        gSystem->Exit(1);
    };

    std::string magic;
    if (!ReadLine(f, magic) || magic != kMagic)
        bail("not a kc761calib export (bad magic line)");
    std::string calibFormula, resolFormula;
    if (!ReadLine(f, calibFormula) || !ReadLine(f, resolFormula))
        bail("truncated header (formula lines missing)");
    StripNewline(calibFormula);
    StripNewline(resolFormula);

    int64_t nCalib = 0;
    double calib[kNC];
    if (!ReadRaw(f, &nCalib, sizeof(nCalib)) ||
        nCalib != kNC ||
        !ReadRaw(f, calib, sizeof(calib)))
        bail("malformed calibration-parameter block");

    int64_t nResol = 0;
    double resol[kNB];
    double resolERef = 0.0;
    if (!ReadRaw(f, &nResol, sizeof(nResol)) ||
        nResol != kNB ||
        !ReadRaw(f, resol, sizeof(resol)) ||
        !ReadRaw(f, &resolERef, sizeof(resolERef)))
        bail("malformed resolution-parameter block");

    int64_t nCore = 0;
    std::vector<double> paramCov(static_cast<size_t>(kNCore) * kNCore);
    if (!ReadRaw(f, &nCore, sizeof(nCore)) ||
        nCore != kNCore ||
        !ReadRaw(f, paramCov.data(), paramCov.size() * sizeof(double)))
        bail("malformed parameter-covariance block");

    int64_t nCh = 0;
    if (!ReadRaw(f, &nCh, sizeof(nCh)) || nCh < 1 || nCh > kMaxChannels)
        bail("malformed channel count");

    // Validate the total size before allocating, so a corrupt header
    // cannot drive a huge allocation.
    const long payloadStart = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    const long fileSize = std::ftell(f);
    std::fseek(f, payloadStart, SEEK_SET);
    const long expected = static_cast<long>(
        8 * (nCh + 1 + 2 * nCh * nCh)); // edges + matrix + matrix_errors
    if (fileSize - payloadStart != expected)
        bail("export size mismatch (truncated or corrupt file)");

    std::vector<double> edges(static_cast<size_t>(nCh) + 1);
    if (!ReadRaw(f, edges.data(), edges.size() * sizeof(double)))
        bail("malformed energy-edge block");
    for (int64_t i = 0; i < nCh; ++i)
        if (!(edges[i] < edges[i + 1]))
            bail("energy edges are not strictly increasing");

    const size_t nEntries = static_cast<size_t>(nCh) * static_cast<size_t>(nCh);
    std::vector<double> matrix(nEntries);
    if (!ReadRaw(f, matrix.data(), matrix.size() * sizeof(double)))
        bail("malformed response-matrix block");
    std::vector<double> matrixErrors(nEntries);
    if (!ReadRaw(f, matrixErrors.data(), matrixErrors.size() * sizeof(double)))
        bail("malformed response-matrix-errors block");
    if (std::fgetc(f) != EOF)
        bail("trailing bytes after the response matrix errors");
    std::fclose(f);
    f = nullptr;

    TFile fout(output.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        std::cerr << "[calib2root] error: cannot create output file: "
                  << output << "\n";
        gSystem->Exit(1);
        return;
    }
    fout.cd();

    // Write order: calibration formula, calibration parameters (each with
    // its 1-sigma error right after it), resolution formula, resolution
    // reference energy, resolution parameters (each with its 1-sigma
    // error), the parameter order and covariance, response matrix with its
    // per-bin errors.  The metadata block is written by the shared helper
    // (the contract read by name in kc761util/calibfile.py and
    // kc761unfold); the per-parameter errors are the covariance diagonals.
    double calibErr[kNC];
    for (int i = 0; i < kNC; ++i)
        calibErr[i] = std::sqrt(std::max(paramCov[i * kNCore + i], 0.0));
    double resolErr[kNB];
    for (int i = 0; i < kNB; ++i) {
        const int diag = (kNC + i) * kNCore + (kNC + i);
        resolErr[i] = std::sqrt(std::max(paramCov[diag], 0.0));
    }
    WriteCalibrationMetadata(calibFormula, calib, calibErr, resolFormula,
                             resolERef, resol, resolErr,
                             "c0 c1 c2 c3 b0 b1 b2", paramCov);

    TH2D* hResp = new TH2D(
        "deposition_response_matrix",
        "KC761 deposition response matrix (energy deposition -> detected"
        " channel);Channel;Energy deposition (keV)",
        static_cast<int>(nCh), -0.5, static_cast<double>(nCh) - 0.5,
        static_cast<int>(nCh), edges.data());
    // The shared filler stores squared errors in fSumw2; the export
    // carries the 1-sigma values.
    std::vector<double> respSw2(nEntries);
    for (size_t k = 0; k < nEntries; ++k)
        respSw2[k] = matrixErrors[k] * matrixErrors[k];
    FillResponseMatrix(hResp, matrix, respSw2, nCh,
                       "deposition_response_matrix");

    hResp->Write();
    fout.Close();

    // Verify the ROOT file was written completely before deleting the only
    // serialized copy of the response: reopen it and require the deposition
    // response matrix with the expected bin counts and its error array.  On
    // any write failure (e.g. a full disk) the export file is kept so the
    // conversion can be re-run.
    TFile fcheck(output.c_str());
    TH2D* hCheck = dynamic_cast<TH2D*>(fcheck.Get("deposition_response_matrix"));
    if (fcheck.IsZombie() || !hCheck ||
        hCheck->GetNbinsX() != static_cast<int>(nCh) ||
        hCheck->GetNbinsY() != static_cast<int>(nCh) ||
        hCheck->GetSumw2() == nullptr) {
        std::cerr << "[calib2root] error: output file is missing or incomplete "
                  << "after writing (response matrix without its error array); "
                  << "keeping the export file: " << exportFile << "\n";
        gSystem->Exit(1);
        return;
    }

    if (gSystem->Unlink(exportFile.c_str()) != 0) {
        std::cerr << "[calib2root] warning: cannot delete the temporary "
                  << "export file: " << exportFile << "\n";
    }

    std::cout << "[calib2root] wrote " << output << " : " << nCh << " x " << nCh
              << " deposition response matrix with per-bin errors, "
              << "calibration/resolution formulas, parameters and their "
              << "covariance\n";
}
