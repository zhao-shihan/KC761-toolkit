// calib2root.cxx
// Convert the kc761calib binary export (written by kc761calib/export.py)
// into the calibration ROOT file.  Internal to kc761calib.
//
// Objects are written in this order:
//   TNamed              "calib_formula"   calibration formula text
//   TParameter<double>  "c0".."c3"        cubic calibration coefficients
//   TNamed              "resol_formula"   resolution formula text
//   TParameter<double>  "resol_e_ref"     resolution reference energy (keV)
//   TParameter<double>  "b0".."b2"        resolution parameters (keV)
//   TH2D                "response_matrix" x = detected channel (uniform
//                       bins of width 1, integer bin centers), y = true
//                       energy (variable-width bins from the calibration
//                       image of the channel bins); content
//                       R[ch, E] = probability that a count in the
//                       true-energy bin E is detected in channel bin ch.
//                       The columns are NOT renormalized: the Gaussian
//                       probability beyond the detector channel range is
//                       truncated, i.e. physically lost.
//
// The temporary export file is deleted after a successful write; on error
// it is left in place so the caller can inspect or re-run it.
//
// Usage:  root -l -b -q 'calib2root.cxx("export.tmp","out.root")'

#include "TFile.h"
#include "TH2D.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TSystem.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kMagic = "kc761calib-export-v1\n";
const int kNC = 4;  // calibration coefficients c0..c3
const int kNB = 3;  // resolution parameters b0..b2
const int64_t kMaxChannels = 1LL << 20;  // sanity cap for the channel count

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
    if (!ReadRaw(f, &nCalib, sizeof(nCalib)) || nCalib != kNC
        || !ReadRaw(f, calib, sizeof(calib)))
        bail("malformed calibration-parameter block");

    int64_t nResol = 0;
    double resol[kNB];
    double resolERef = 0.0;
    if (!ReadRaw(f, &nResol, sizeof(nResol)) || nResol != kNB
        || !ReadRaw(f, resol, sizeof(resol))
        || !ReadRaw(f, &resolERef, sizeof(resolERef)))
        bail("malformed resolution-parameter block");

    int64_t nCh = 0;
    if (!ReadRaw(f, &nCh, sizeof(nCh)) || nCh < 1 || nCh > kMaxChannels)
        bail("malformed channel count");

    // Validate the total size before allocating, so a corrupt header
    // cannot drive a huge allocation.
    const long payloadStart = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    const long fileSize = std::ftell(f);
    std::fseek(f, payloadStart, SEEK_SET);
    const long expected = static_cast<long>(8 * (nCh + 1 + nCh * nCh));
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
    if (std::fgetc(f) != EOF)
        bail("trailing bytes after the response matrix");
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

    // Write order: calibration formula, calibration parameters, resolution
    // formula, resolution reference energy, resolution parameters, response
    // matrix.
    (new TNamed("calib_formula", calibFormula.c_str()))->Write();
    const char* kCalibNames[kNC] = {"c0", "c1", "c2", "c3"};
    for (int i = 0; i < kNC; ++i)
        (new TParameter<double>(kCalibNames[i], calib[i]))->Write();
    (new TNamed("resol_formula", resolFormula.c_str()))->Write();
    (new TParameter<double>("resol_e_ref", resolERef))->Write();
    const char* kResolNames[kNB] = {"b0", "b1", "b2"};
    for (int i = 0; i < kNB; ++i)
        (new TParameter<double>(kResolNames[i], resol[i]))->Write();

    TH2D* hResp = new TH2D(
        "response_matrix",
        "KC761 response matrix (true energy -> detected channel);"
        "Channel;True energy (keV)",
        static_cast<int>(nCh), -0.5, static_cast<double>(nCh) - 0.5,
        static_cast<int>(nCh), edges.data());

    // TH2 linearizes the content array as binx + (nbinsx + 2)*biny, so the
    // bin (x = ch + 1, y = E + 1) sits at (E + 1)*(nCh + 2) + (ch + 1).
    // matrix is row-major with row = channel and column = true-energy bin.
    // TH2D stores the contents in the inherited TArrayD, exposed by
    // GetArray() as a Double_t*.
    Double_t* arr = hResp->GetArray();
    const int stride = static_cast<int>(nCh) + 2;
    for (int64_t j = 0; j < nCh; ++j) {            // y: true-energy bin
        Double_t* dst = &arr[(j + 1) * stride + 1];  // x: channel bins 0..nCh-1
        for (int64_t i = 0; i < nCh; ++i)
            dst[i] = matrix[i * nCh + j];
    }
    // Guard the array-layout assumption against future ROOT changes: the
    // four corners must agree with the file's row-major layout.  The fill
    // writes bin (x = ch + 1, y = E + 1) = matrix[ch*nCh + E], so
    // GetBinContent(1, nCh) holds matrix[nCh - 1] (channel 0, energy
    // nCh-1) and GetBinContent(nCh, 1) holds matrix[(nCh-1)*nCh] (channel
    // nCh-1, energy 0).
    if (hResp->GetBinContent(1, 1) != matrix[0]
        || hResp->GetBinContent(1, static_cast<int>(nCh)) != matrix[nCh - 1]
        || hResp->GetBinContent(static_cast<int>(nCh), 1)
               != matrix[static_cast<size_t>(nCh) * (nCh - 1)]
        || hResp->GetBinContent(static_cast<int>(nCh), static_cast<int>(nCh))
               != matrix[nEntries - 1]) {
        std::cerr << "[calib2root] error: ROOT bin-layout assumption violated; "
                  << "the response matrix would be transposed\n";
        gSystem->Exit(1);
        return;
    }

    hResp->Write();
    fout.Close();

    // Verify the ROOT file was written completely before deleting the only
    // serialized copy of the response: reopen it and require the response
    // matrix with the expected bin counts.  On any write failure (e.g. a
    // full disk) the export file is kept so the conversion can be re-run.
    TFile fcheck(output.c_str());
    TH2D* hCheck = dynamic_cast<TH2D*>(fcheck.Get("response_matrix"));
    if (fcheck.IsZombie() || !hCheck
        || hCheck->GetNbinsX() != static_cast<int>(nCh)
        || hCheck->GetNbinsY() != static_cast<int>(nCh)) {
        std::cerr << "[calib2root] error: output file is missing or incomplete "
                  << "after writing; keeping the export file: " << exportFile
                  << "\n";
        gSystem->Exit(1);
        return;
    }

    if (gSystem->Unlink(exportFile.c_str()) != 0) {
        std::cerr << "[calib2root] warning: cannot delete the temporary "
                  << "export file: " << exportFile << "\n";
    }

    std::cout << "[calib2root] wrote " << output << " : " << nCh << " x " << nCh
              << " response matrix, calibration/resolution formulas and "
              << "parameters\n";
}
