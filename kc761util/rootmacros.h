// rootmacros.h
// Shared helpers for the toolkit's standalone ROOT writer macros
// (calib2root.cxx, sim2root.cxx, unfold2root.cxx): binary-export reading and the
// guarded row-major TH2 fill.  Each macro is invoked by root in its own
// process, so plain (non-static) helpers cannot collide between macros.
//
// Included as `#include "../kc761util/rootmacros.h"` from the macro files;
// cling resolves quoted includes relative to the including file.

#ifndef KC761UTIL_ROOTMACROS_H
#define KC761UTIL_ROOTMACROS_H

#include "TFile.h"
#include "TH2D.h"
#include "TMatrixDSym.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TSystem.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Read one line of text (up to and including the newline); returns true
// while there is content (a trailing line without a newline is accepted).
inline bool ReadLine(std::FILE* f, std::string& line) {
    line.clear();
    for (;;) {
        int c = std::fgetc(f);
        if (c == EOF) return !line.empty();
        line.push_back(static_cast<char>(c));
        if (c == '\n') return true;
    }
}

inline bool ReadRaw(std::FILE* f, void* buf, size_t nbytes) {
    return std::fread(buf, 1, nbytes, f) == nbytes;
}

// Abort the macro with a diagnostic (the standalone writers' single
// failure path).
inline void Fatal(const std::string& message) {
    std::cerr << "error: " << message << "\n";
    gSystem->Exit(1);
}

inline void StripNewline(std::string& s) {
    if (!s.empty() && s.back() == '\n') s.pop_back();
}

// NaN-tolerant equality: inherited uncertainty arrays may carry NaN
// entries when a parameter is undetermined.
inline bool SameValue(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b;
}

// Write the inherited calibration-metadata objects in the kc761calib
// order -- the contract read by name in kc761util/calibfile.py and
// kc761unfold: calib_formula, c0..c3 (each followed by its _err),
// resol_formula, resol_e_ref, b0..b2 (each followed by its _err),
// param_order, and the 7x7 param_cov in the param_order basis.
inline void WriteCalibrationMetadata(
    const std::string& calibFormula, const double* calib,
    const double* calibErr, const std::string& resolFormula,
    double resolERef, const double* resol, const double* resolErr,
    const std::string& paramOrder, const std::vector<double>& paramCov) {
    (new TNamed("calib_formula", calibFormula.c_str()))->Write();
    const char* kCalibNames[4] = {"c0", "c1", "c2", "c3"};
    for (int i = 0; i < 4; ++i) {
        (new TParameter<double>(kCalibNames[i], calib[i]))->Write();
        const std::string errName = std::string(kCalibNames[i]) + "_err";
        (new TParameter<double>(errName.c_str(), calibErr[i]))->Write();
    }
    (new TNamed("resol_formula", resolFormula.c_str()))->Write();
    (new TParameter<double>("resol_e_ref", resolERef))->Write();
    const char* kResolNames[3] = {"b0", "b1", "b2"};
    for (int i = 0; i < 3; ++i) {
        (new TParameter<double>(kResolNames[i], resol[i]))->Write();
        const std::string errName = std::string(kResolNames[i]) + "_err";
        (new TParameter<double>(errName.c_str(), resolErr[i]))->Write();
    }
    // Format marker: 3 = the exported matrix uses the fit's convention
    // (5-sigma kernel-support cutoff + column renormalization).  Files
    // without this marker predate that convention and must not be
    // unfolded against (their column sums are not 1).
    (new TNamed("format_version", "3"))->Write();
    (new TNamed("param_order", paramOrder.c_str()))->Write();
    TMatrixDSym* cov = new TMatrixDSym(7);
    for (int r = 0; r < 7; ++r)
        for (int c = 0; c < 7; ++c)
            (*cov)(r, c) = paramCov[r * 7 + c];
    cov->Write("param_cov");
}

// Fill a TH2D's content and uncertainty arrays from row-major (nCh x nCh)
// vectors (row = x bin, column = y bin) and guard the array-layout
// assumption.  TH2 linearizes the content array as
// binx + (nbinsx + 2)*biny, so the bin (x = ch + 1, y = e + 1) sits at
// (e + 1)*(nCh + 2) + (ch + 1); fSumw2 (allocated by Sumw2()) uses the
// same linearization and holds the squared uncertainties.
inline void FillMatrix(TH2D* h, const std::vector<double>& m,
                       const std::vector<double>& sw2, int64_t nCh,
                       const char* name) {
    h->Sumw2();
    Double_t* arr = h->GetArray();
    Double_t* sumw2 = h->GetSumw2()->GetArray();
    const int stride = static_cast<int>(nCh) + 2;
    for (int64_t j = 0; j < nCh; ++j) { // y: energy bin
        const int base = static_cast<int>(j + 1) * stride + 1;
        for (int64_t i = 0; i < nCh; ++i) { // x: channel bin
            const int bin = base + static_cast<int>(i);
            const size_t off = static_cast<size_t>(i) * nCh + j;
            arr[bin] = m[off];
            sumw2[bin] = sw2[off];
        }
    }
    const size_t nEntries = static_cast<size_t>(nCh) * static_cast<size_t>(nCh);
    const auto sw2Cell = [&](int64_t ch, int64_t e) {
        return sumw2[(e + 1) * stride + (ch + 1)];
    };
    if (!SameValue(h->GetBinContent(1, 1), m[0]) ||
        !SameValue(h->GetBinContent(1, static_cast<int>(nCh)), m[nCh - 1]) ||
        !SameValue(h->GetBinContent(static_cast<int>(nCh), 1),
                   m[static_cast<size_t>(nCh) * (nCh - 1)]) ||
        !SameValue(h->GetBinContent(static_cast<int>(nCh),
                                    static_cast<int>(nCh)),
                   m[nEntries - 1]) ||
        !SameValue(sw2Cell(0, 0), sw2[0]) ||
        !SameValue(sw2Cell(0, nCh - 1), sw2[nCh - 1]) ||
        !SameValue(sw2Cell(nCh - 1, 0),
                   sw2[static_cast<size_t>(nCh) * (nCh - 1)]) ||
        !SameValue(sw2Cell(nCh - 1, nCh - 1), sw2[nEntries - 1])) {
        std::cerr << "error: ROOT bin-layout assumption violated; " << name
                  << " would be transposed\n";
        gSystem->Exit(1);
    }
}

#endif // KC761UTIL_ROOTMACROS_H
