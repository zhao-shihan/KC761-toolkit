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
//   TH2D                "response_matrix" x = detected channel (uniform
//                       bins of width 1, integer bin centers), y = true
//                       energy (variable-width bins from the calibration
//                       image of the channel bins); content
//                       R[ch, E] = probability that a count in the
//                       true-energy bin E is detected in channel bin ch.
//                       The columns are NOT renormalized: the Gaussian
//                       probability beyond the detector channel range is
//                       truncated, i.e. physically lost.  The per-bin
//                       errors (fSumw2) hold the per-element 1-sigma
//                       uncertainty propagated linearly from param_cov.
//
// The temporary export file is deleted after a successful write; on error
// it is left in place so the caller can inspect or re-run it.
//
// Usage:  root -l -b -q 'calib2root.cxx("export.tmp","out.root")'

#include "TFile.h"
#include "TH2D.h"
#include "TMatrixDSym.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TSystem.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kMagic = "kc761calib-export-v2\n";
const int kNC = 4;                    // calibration coefficients c0..c3
const int kNB = 3;                    // resolution parameters b0..b2
const int kNCore = 7;                 // stored parameters (c0..c3, b0..b2)
const int64_t kMaxChannels = 1 << 14; // Sanity cap for the channel count. 8 * 16384^2 = 2 GiB per block

// Read one '\n'-terminated text line (the magic and the two formula
// lines at the head of the export file).
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

void StripNewline(std::string& s) {
    if (!s.empty() && s.back() == '\n') s.pop_back();
}

bool SameValue(double a, double b) {
    // NaN-tolerant equality: the error matrix may carry NaN entries when a
    // parameter is undetermined, and the layout guard must not trip on them.
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b;
}

// Fill a TH2D's content and error arrays from row-major (nCh x nCh)
// vectors (row = channel bin, column = true-energy bin) and guard the
// array-layout assumption.
void FillResponseMatrix(TH2D* h, const std::vector<double>& m,
                        const std::vector<double>& errs, int64_t nCh,
                        const char* name) {
    // TH2 linearizes the content array as binx + (nbinsx + 2)*biny, so the
    // bin (x = ch + 1, y = E + 1) sits at (E + 1)*(nCh + 2) + (ch + 1).
    // m and errs are row-major with row = channel and column = true-energy
    // bin.  TH2D stores the contents in the inherited TArrayD exposed by
    // GetArray(); the bin errors live in the fSumw2 array (allocated by
    // Sumw2()) in the same linearization, holding the squared errors.
    h->Sumw2();
    Double_t* arr = h->GetArray();
    Double_t* sw2 = h->GetSumw2()->GetArray();
    const int stride = static_cast<int>(nCh) + 2;
    for (int64_t j = 0; j < nCh; ++j) { // y: true-energy bin
        const int base = static_cast<int>(j + 1) * stride + 1;
        for (int64_t i = 0; i < nCh; ++i) { // x: channel bin
            const int bin = base + static_cast<int>(i);
            const size_t off = static_cast<size_t>(i) * nCh + j;
            arr[bin] = m[off];
            sw2[bin] = errs[off] * errs[off];
        }
    }
    // Guard the array-layout assumption against future ROOT changes: the
    // four corners must agree with the file's row-major layout.  The fill
    // writes bin (x = ch + 1, y = E + 1) = m[ch*nCh + E], so
    // GetBinContent(1, nCh) holds m[nCh - 1] (channel 0, energy nCh-1)
    // and GetBinContent(nCh, 1) holds m[(nCh-1)*nCh] (channel nCh-1,
    // energy 0); the fSumw2 cells hold the corresponding squared errors.
    const size_t nEntries = static_cast<size_t>(nCh) * static_cast<size_t>(nCh);
    const auto sw2Cell = [&](int64_t ch, int64_t e) {
        return sw2[(e + 1) * stride + (ch + 1)];
    };
    if (!SameValue(h->GetBinContent(1, 1), m[0]) ||
        !SameValue(h->GetBinContent(1, static_cast<int>(nCh)), m[nCh - 1]) ||
        !SameValue(h->GetBinContent(static_cast<int>(nCh), 1),
                   m[static_cast<size_t>(nCh) * (nCh - 1)]) ||
        !SameValue(h->GetBinContent(static_cast<int>(nCh),
                                    static_cast<int>(nCh)),
                   m[nEntries - 1]) ||
        !SameValue(sw2Cell(0, 0), errs[0] * errs[0]) ||
        !SameValue(sw2Cell(0, nCh - 1), errs[nCh - 1] * errs[nCh - 1]) ||
        !SameValue(sw2Cell(nCh - 1, 0),
                   errs[static_cast<size_t>(nCh) * (nCh - 1)] * errs[static_cast<size_t>(nCh) * (nCh - 1)]) ||
        !SameValue(sw2Cell(nCh - 1, nCh - 1),
                   errs[nEntries - 1] * errs[nEntries - 1])) {
        std::cerr << "[calib2root] error: ROOT bin-layout assumption violated; "
                  << name << " would be transposed\n";
        gSystem->Exit(1);
    }
}

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
    // per-bin errors.
    (new TNamed("calib_formula", calibFormula.c_str()))->Write();
    const char* kCalibNames[kNC] = {"c0", "c1", "c2", "c3"};
    for (int i = 0; i < kNC; ++i) {
        (new TParameter<double>(kCalibNames[i], calib[i]))->Write();
        const double err = std::sqrt(std::max(paramCov[i * kNCore + i], 0.0));
        const std::string errName = std::string(kCalibNames[i]) + "_err";
        (new TParameter<double>(errName.c_str(), err))->Write();
    }
    (new TNamed("resol_formula", resolFormula.c_str()))->Write();
    (new TParameter<double>("resol_e_ref", resolERef))->Write();
    const char* kResolNames[kNB] = {"b0", "b1", "b2"};
    for (int i = 0; i < kNB; ++i) {
        (new TParameter<double>(kResolNames[i], resol[i]))->Write();
        const int diag = (kNC + i) * kNCore + (kNC + i);
        const double err = std::sqrt(std::max(paramCov[diag], 0.0));
        const std::string errName = std::string(kResolNames[i]) + "_err";
        (new TParameter<double>(errName.c_str(), err))->Write();
    }

    (new TNamed("param_order", "c0 c1 c2 c3 b0 b1 b2"))->Write();
    TMatrixDSym* cov = new TMatrixDSym(kNCore);
    for (int r = 0; r < kNCore; ++r)
        for (int c = 0; c < kNCore; ++c)
            (*cov)(r, c) = paramCov[r * kNCore + c];
    cov->Write("param_cov");

    TH2D* hResp = new TH2D(
        "response_matrix",
        "KC761 response matrix (true energy -> detected channel) [bin errors"
        " = propagated 1-sigma fit uncertainty];Channel;True energy (keV)",
        static_cast<int>(nCh), -0.5, static_cast<double>(nCh) - 0.5,
        static_cast<int>(nCh), edges.data());
    FillResponseMatrix(hResp, matrix, matrixErrors, nCh, "response_matrix");

    hResp->Write();
    fout.Close();

    // Verify the ROOT file was written completely before deleting the only
    // serialized copy of the response: reopen it and require the response
    // matrix with the expected bin counts and its error array.  On any
    // write failure (e.g. a full disk) the export file is kept so the
    // conversion can be re-run.
    TFile fcheck(output.c_str());
    TH2D* hCheck = dynamic_cast<TH2D*>(fcheck.Get("response_matrix"));
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
              << " response matrix with per-bin errors, "
              << "calibration/resolution formulas, parameters and their "
              << "covariance\n";
}
