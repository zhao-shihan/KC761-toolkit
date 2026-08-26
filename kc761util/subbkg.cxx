// subbkg.cxx
// ---------------------------------------------------------------------------
// Subtract a background spectrum from a signal spectrum, scaling the
// background by the ratio of acquisition (DAQ) times, with proper Poisson
// error propagation.
//
// Input ROOT files (both as produced by csv2root.cxx):
//   - TH1D              "kc761_spectrum" : experimental spectrum
//   - TParameter<double> "daq_time"      : acquisition time in hours
//
// For each bin:
//     r      = t_signal / t_background
//     net_i  = S_i - r * B_i
//     err_i  = sqrt(S_i + r^2 * B_i)      (Poisson statistics on both inputs)
//
// This is implemented with the canonical TH1 operations:
//   - Sumw2() stores the Poisson variance per bin (var == counts) in the
//     histograms' fSumw2 arrays;
//   - TH1::Add(hSig, hBkg, 1.0, -r) computes net = 1*S - r*B in one step,
//     propagating the errors as var_net = 1^2*S + r^2*B.
//
// Output ROOT file contains:
//   - TH1D "kc761_spectrum" : background-subtracted spectrum
//
// Usage:
//   root -l -b -q 'subbkg.cxx("sig.root","bkg.root")'
//   root -l -b -q 'subbkg.cxx("sig.root","bkg.root","out.root")'
// ---------------------------------------------------------------------------

#include "TFile.h"
#include "TH1D.h"
#include "TParameter.h"
#include "TSystem.h"

#include <iostream>
#include <string>

void subbkg(const std::string& sigFile, const std::string& bkgFile,
            const std::string& outFile = "") {
    // Default output name: signal filename with "_subbkg" before the extension.
    std::string outName = outFile;
    if (outName.empty()) {
        outName = sigFile;
        size_t dot = outName.rfind('.');
        if (dot != std::string::npos) outName = outName.substr(0, dot);
        outName += "_subbkg.root";
    }

    TFile* fSig = TFile::Open(sigFile.c_str());
    if (!fSig || fSig->IsZombie()) {
        std::cerr << "[subbkg] error: cannot open signal file: " << sigFile << "\n";
        gSystem->Exit(1);
        return;
    }
    TFile* fBkg = TFile::Open(bkgFile.c_str());
    if (!fBkg || fBkg->IsZombie()) {
        std::cerr << "[subbkg] error: cannot open background file: " << bkgFile << "\n";
        gSystem->Exit(1);
        return;
    }

    TH1D* hSig = dynamic_cast<TH1D*>(fSig->Get("kc761_spectrum"));
    TH1D* hBkg = dynamic_cast<TH1D*>(fBkg->Get("kc761_spectrum"));
    TParameter<double>* tSig = dynamic_cast<TParameter<double>*>(fSig->Get("daq_time"));
    TParameter<double>* tBkg = dynamic_cast<TParameter<double>*>(fBkg->Get("daq_time"));

    if (!hSig || !hBkg || !tSig || !tBkg) {
        std::cerr << "[subbkg] error: files must contain TH1D \"kc761_spectrum\" and "
                  << "TParameter<double> \"daq_time\"\n";
        gSystem->Exit(1);
        return;
    }

    int nBins = hSig->GetNbinsX();
    if (hBkg->GetNbinsX() != nBins ||
        hSig->GetXaxis()->GetXmin() != hBkg->GetXaxis()->GetXmin() ||
        hSig->GetXaxis()->GetXmax() != hBkg->GetXaxis()->GetXmax()) {
        std::cerr << "[subbkg] error: signal/background binning mismatch ("
                  << nBins << " bins vs " << hBkg->GetNbinsX() << " bins)\n";
        gSystem->Exit(1);
        return;
    }

    double tBkgVal = tBkg->GetVal();
    if (tBkgVal <= 0.0) {
        std::cerr << "[subbkg] error: background daq_time must be > 0 (got "
                  << tBkgVal << ")\n";
        gSystem->Exit(1);
        return;
    }
    double r = tSig->GetVal() / tBkgVal;

    // Poisson statistics: variance per bin == number of counts.  Storing this
    // in fSumw2 lets TH1::Add propagate the errors correctly.
    hSig->Sumw2();
    hBkg->Sumw2();

    // Open the output file first and make it current, so hNet is created in
    // its directory (avoids clashing with the input histograms by name).
    TFile fOut(outName.c_str(), "RECREATE");
    fOut.cd();

    // Clone the signal histogram (keeps binning/axes and the fSumw2 array).
    TH1D* hNet = new TH1D(*hSig);
    hNet->SetNameTitle("kc761_spectrum",
                       "kc761 spectrum (background subtracted);Channel;Counts");
    // net = 1*S - r*B, with var_net = 1^2*var_S + r^2*var_B = S + r^2*B.
    hNet->Add(hSig, hBkg, 1.0, -r);

    hNet->Write();
    fOut.Close();

    std::cout << "[subbkg] wrote " << outName << " : " << nBins
              << " bins, scale r = t_sig/t_bkg = " << r << "\n";
}
