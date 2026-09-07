// matrix2root.cxx
// Convert the kc761sim matrix-mode binary export (written by
// kc761sim/compose.py) into the composite ROOT file.  Internal
// to the kc761sim matrix modes.
//
// The export carries three matrices with the toolkit conventions
// (matrix[i, j] = probability that a count in the input-side bin j is
// detected in the output-side bin i; columns are NOT renormalized, so
// column sums below 1 are the physical detection efficiency) and their
// per-element 1-sigma squared errors for fSumw2.  The x axis of every
// matrix is its output side and the y axis its input side:
//
//   TH2D  "deposition_to_channel"  C copy: x = detected channel
//         (uniform bins of width 1), y = energy deposition (variable
//         bins, the calibration image of the channels).  Inherited
//         bitwise from the input calibration file.
//   TH2D  "primary_to_deposition"   G: x = energy deposition, y =
//         primary energy (both axes variable, identical edges), MC
//         counts with statistical errors.
//   TH2D  "primary_to_channel"             R = C @ G: x = detected channel
//         (uniform bins of width 1), y = true primary gamma energy
//         (variable bins), fSumw2 = propagated per-element 1-sigma
//         variance (calibration fit covariance via the analytic Jacobian
//         plus the Monte Carlo multinomial covariance).
//   TH1D  "detection_efficiency"          detection efficiency per
//         primary-energy bin with propagated errors.
//
// The zero-deposition counts are NOT stored: they are recovered as
// n_b - sum_dep G[dep, b] with the per-column primary totals n_b fixed by
// the simulation scheme (see compose.py); they enter the composition
// through the normalization totals and the R errors.
//
// Inherited calibration parameters are written first, exactly as
// kc761calib/calib2root.cxx writes them (formulas, c0..c3 +/- errors,
// resol_e_ref, b0..b2 +/- errors, param_order, the 7x7 param_cov), so the
// file is a drop-in replacement for kc761unfold.  The new mode parameters
// (simulation_mode, simulation_mode_name, geometry, angular distribution,
// n_events, seed, input path and checksum) follow the matrices.
//
// The temporary export file is deleted after a successful write; on error
// it is left in place so the caller can inspect or re-run it.
//
// Usage:  root -l -b -q 'matrix2root.cxx("export.tmp","out.root")'
#include "../kc761util/rootmacros.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TSystem.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kMagic = "kc761sim-matrix-export-v1\n";
const int kNC = 4;                    // calibration coefficients c0..c3
const int kNB = 3;                    // resolution parameters b0..b2
const int kNCore = 7;                 // stored parameters (c0..c3, b0..b2)
const int64_t kMaxChannels = 1 << 14; // sanity cap (2 GiB per block)

// Fill a TH1D's content and squared-error arrays (same linearization as
// FillMatrix, one axis).
void FillH1(TH1D* h, const std::vector<double>& m,
            const std::vector<double>& sw2, int64_t nCh, const char* name) {
    h->Sumw2();
    Double_t* arr = h->GetArray();
    Double_t* sumw2 = h->GetSumw2()->GetArray();
    for (int64_t i = 0; i < nCh; ++i) {
        arr[i + 1] = m[static_cast<size_t>(i)];
        sumw2[i + 1] = sw2[static_cast<size_t>(i)];
    }
    if (!SameValue(h->GetBinContent(1), m[0]) ||
        !SameValue(h->GetBinContent(static_cast<int>(nCh)),
                   m[static_cast<size_t>(nCh) - 1])) {
        std::cerr << "[matrix2root] error: ROOT bin-layout assumption "
                  << "violated; " << name << " misplaced\n";
        gSystem->Exit(1);
    }
}

} // namespace

void matrix2root(const std::string& exportFile, const std::string& output) {
    std::FILE* f = std::fopen(exportFile.c_str(), "rb");
    if (!f) {
        std::cerr << "[matrix2root] error: cannot open export file: "
                  << exportFile << "\n";
        gSystem->Exit(1);
        return;
    }
    const auto bail = [&](const std::string& message) {
        std::cerr << "[matrix2root] error: " << message << "\n";
        std::fclose(f);
        gSystem->Exit(1);
    };

    std::string magic;
    if (!ReadLine(f, magic) || magic != kMagic)
        bail("not a kc761sim matrix export (bad magic line)");

    std::string modeName, geometryName, angular;
    std::string inputPath, inputSha, calibFormula;
    std::string resolFormula, paramOrder;
    if (!ReadLine(f, modeName) || !ReadLine(f, geometryName) ||
        !ReadLine(f, angular) || !ReadLine(f, inputPath) ||
        !ReadLine(f, inputSha) || !ReadLine(f, calibFormula) ||
        !ReadLine(f, resolFormula) || !ReadLine(f, paramOrder))
        bail("truncated header (text lines missing)");
    StripNewline(modeName);
    StripNewline(geometryName);
    StripNewline(angular);
    StripNewline(inputPath);
    StripNewline(inputSha);
    StripNewline(calibFormula);
    StripNewline(resolFormula);
    StripNewline(paramOrder);

    int64_t nCh = 0;
    int64_t mode = 0;
    if (!ReadRaw(f, &nCh, sizeof(nCh)) || nCh < 1 || nCh > kMaxChannels ||
        !ReadRaw(f, &mode, sizeof(mode)) || (mode != 1 && mode != 2))
        bail("malformed size/mode block");

    std::vector<double> edges(static_cast<size_t>(nCh) + 1);
    if (!ReadRaw(f, edges.data(), edges.size() * sizeof(double)))
        bail("malformed edge block");
    for (int64_t i = 0; i < nCh; ++i)
        if (!(edges[i] < edges[i + 1]))
            bail("energy edges are not strictly increasing");

    double geometryParam = 0.0;
    double nEvents = 0.0, seed = 0.0;
    if (!ReadRaw(f, &geometryParam, sizeof(geometryParam)) ||
        !ReadRaw(f, &nEvents, sizeof(nEvents)) ||
        !ReadRaw(f, &seed, sizeof(seed)))
        bail("malformed geometry/run-parameter block");

    double calib[kNC];
    double calibErr[kNC];
    double resol[kNB];
    double resolErr[kNB];
    double resolERef = 0.0;
    if (!ReadRaw(f, calib, sizeof(calib)) ||
        !ReadRaw(f, calibErr, sizeof(calibErr)) ||
        !ReadRaw(f, resol, sizeof(resol)) ||
        !ReadRaw(f, resolErr, sizeof(resolErr)) ||
        !ReadRaw(f, &resolERef, sizeof(resolERef)))
        bail("malformed calibration-parameter block");

    std::vector<double> paramCov(static_cast<size_t>(kNCore) * kNCore);
    if (!ReadRaw(f, paramCov.data(), paramCov.size() * sizeof(double)))
        bail("malformed parameter-covariance block");

    // Validate the remaining payload size before allocating: eight
    // tagged blocks (each with an int64 id) -- six nCh x nCh blocks
    // (C, C_err, G, G_sumw2, R, R_var) plus two nCh blocks (efficiency,
    // efficiency_var).
    const long payloadStart = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    const long fileSize = std::ftell(f);
    std::fseek(f, payloadStart, SEEK_SET);
    const long expected = static_cast<long>(8 * (8 + 6 * nCh * nCh + 2 * nCh));
    if (fileSize - payloadStart != expected)
        bail("export size mismatch (truncated or corrupt file)");

    // Read one tagged payload block; the id must match the expected one,
    // so a layout drift between the writer and the reader fails loudly
    // instead of silently swapping same-shaped blocks.
    const auto readBlock = [&](const std::vector<double>& values,
                               int64_t expectedId, const char* what) {
        int64_t id = 0;
        if (!ReadRaw(f, &id, sizeof(id)) || id != expectedId)
            bail(std::string("block id mismatch for ") + what +
                 " (layout drift or corrupt file)");
        if (!ReadRaw(f, const_cast<double*>(values.data()),
                     values.size() * sizeof(double)))
            bail(std::string("malformed ") + what + " block");
    };

    const size_t nEntries = static_cast<size_t>(nCh) * static_cast<size_t>(nCh);
    std::vector<double> matrix(nEntries);
    std::vector<double> matrixErrors(nEntries);
    readBlock(matrix, 1, "deposition-to-channel");
    readBlock(matrixErrors, 2, "deposition-to-channel errors");
    std::vector<double> gCounts(nEntries);
    std::vector<double> gSumw2(nEntries);
    readBlock(gCounts, 3, "G counts");
    readBlock(gSumw2, 4, "G sumw2");
    std::vector<double> resp(nEntries);
    std::vector<double> respVar(nEntries);
    readBlock(resp, 5, "primary-to-channel");
    readBlock(respVar, 6, "primary-to-channel variance");
    std::vector<double> efficiency(static_cast<size_t>(nCh));
    std::vector<double> efficiencyVar(static_cast<size_t>(nCh));
    readBlock(efficiency, 7, "efficiency");
    readBlock(efficiencyVar, 8, "efficiency variance");
    if (std::fgetc(f) != EOF)
        bail("trailing bytes after the efficiency block");
    std::fclose(f);
    f = nullptr;

    // Write to a temporary sibling path and rename onto the target only
    // after the post-write verification passes, so a failed export (e.g.
    // a full disk) can never destroy a previously good composite or leave
    // a truncated file at the production path.
    const std::string tmpOut = output + ".tmp";
    TFile fout(tmpOut.c_str(), "RECREATE");
    if (fout.IsZombie()) {
        std::cerr << "[matrix2root] error: cannot create temporary output "
                  << "file: " << tmpOut << "\n";
        gSystem->Exit(1);
        return;
    }
    fout.cd();

    // Inherited calibration parameters, written by the shared helper in
    // the kc761calib order -- the contract read by name in
    // kc761util/calibfile.py and kc761unfold.
    WriteCalibrationMetadata(calibFormula, calib, calibErr, resolFormula,
                             resolERef, resol, resolErr, paramOrder,
                             paramCov);

    // C copy: inherited bitwise (content and its own squared errors).
    TH2D* hC = new TH2D(
        "deposition_to_channel",
        "KC761 deposition-to-channel matrix;Channel;Energy deposition (keV)"
        " channel);Channel;Energy deposition (keV)",
        static_cast<int>(nCh), -0.5, static_cast<double>(nCh) - 0.5,
        static_cast<int>(nCh), edges.data());
    std::vector<double> cSw2(nEntries);
    for (size_t k = 0; k < nEntries; ++k)
        cSw2[k] = matrixErrors[k] * matrixErrors[k];
    FillMatrix(hC, matrix, cSw2, nCh, "deposition_to_channel");
    hC->Write();

    // G: Monte Carlo counts with statistical errors (unit-weight fills:
    // the stored sumw2 buffer equals the counts).  x = energy deposition
    // (output side), y = primary energy (input side).
    TH2D* hG = new TH2D(
        "primary_to_deposition",
        "KC761 primary-to-deposition matrix;Energy deposition (keV);Primary gamma"
        " deposition);Energy deposition (keV);Primary gamma energy (keV)",
        static_cast<int>(nCh), edges.data(),
        static_cast<int>(nCh), edges.data());
    FillMatrix(hG, gCounts, gSumw2, nCh, "primary_to_deposition");
    hG->Write();

    // R = C @ G: the primary-to-channel matrix (the only object kc761unfold
    // reads), fSumw2 = propagated per-element 1-sigma variance.
    TH2D* hResp = new TH2D(
        "primary_to_channel",
        "KC761 primary-to-channel matrix;Channel;Primary gamma energy (keV)"
        ";Channel;Primary gamma energy (keV)",
        static_cast<int>(nCh), -0.5, static_cast<double>(nCh) - 0.5,
        static_cast<int>(nCh), edges.data());
    FillMatrix(hResp, resp, respVar, nCh, "primary_to_channel");
    hResp->Write();

    TH1D* hEff = new TH1D(
        "detection_efficiency",
        "Primary detection efficiency;Primary gamma energy (keV)"
        " (keV)",
        static_cast<int>(nCh), edges.data());
    FillH1(hEff, efficiency, efficiencyVar, nCh, "detection_efficiency");
    hEff->Write();

    // New mode parameters (Q30 list).
    (new TParameter<double>("simulation_mode",
                            static_cast<double>(mode)))
        ->Write();
    (new TNamed("simulation_mode_name", modeName.c_str()))->Write();
    (new TParameter<double>(geometryName.c_str(), geometryParam))->Write();
    (new TNamed("angular_distribution", angular.c_str()))->Write();
    (new TParameter<double>("n_events", nEvents))->Write();
    (new TParameter<double>("seed", seed))->Write();
    (new TNamed("input_calib_path", inputPath.c_str()))->Write();
    (new TNamed("input_calib_sha256", inputSha.c_str()))->Write();

    fout.Close();

    // Verify the temporary ROOT file was written completely before it
    // becomes the production output: require the primary-to-channel matrix with the
    // expected bin counts and its error array (the kc761calib
    // convention).  Only after this passes is the file renamed onto the
    // target; on failure the partial file is removed and the export kept.
    TFile fcheck(tmpOut.c_str());
    TH2D* hCheck = dynamic_cast<TH2D*>(fcheck.Get("primary_to_channel"));
    if (fcheck.IsZombie() || !hCheck ||
        hCheck->GetNbinsX() != static_cast<int>(nCh) ||
        hCheck->GetNbinsY() != static_cast<int>(nCh) ||
        hCheck->GetSumw2() == nullptr) {
        fcheck.Close();
        gSystem->Unlink(tmpOut.c_str());
        std::cerr << "[matrix2root] error: temporary output is missing or "
                  << "incomplete after writing (primary-to-channel matrix without "
                  << "its error array); keeping the export file: "
                  << exportFile << "\n";
        gSystem->Exit(1);
        return;
    }
    fcheck.Close();

    if (gSystem->Rename(tmpOut.c_str(), output.c_str()) != 0) {
        gSystem->Unlink(tmpOut.c_str());
        std::cerr << "[matrix2root] error: cannot rename the verified "
                  << "output onto " << output << "; keeping the export "
                  << "file: " << exportFile << "\n";
        gSystem->Exit(1);
        return;
    }

    if (gSystem->Unlink(exportFile.c_str()) != 0) {
        std::cerr << "[matrix2root] warning: cannot delete the temporary "
                  << "export file: " << exportFile << "\n";
    }

    std::cout << "[matrix2root] wrote " << output << " : " << nCh << " x "
              << nCh << " primary-to-channel matrix with per-bin errors, "
              << "deposition-to-channel copy, G counts and parameters\n";
}
