// subbkg.cxx
// Background-subtract a signal spectrum by the ratio of DAQ times,
// propagating Poisson errors: net_i = S_i - r*B_i, err_i = sqrt(S_i + r^2*B_i).
// Usage:  root -l -b -q 'subbkg.cxx("sig.root","bkg.root","out.root")'

#include "TFile.h"
#include "TH1D.h"
#include "TParameter.h"
#include "TSystem.h"

#include <iostream>
#include <string>

// Toolkit-wide histogram-name convention; keep in sync with
// kc761util/spectrum.py (SPECTRUM_HIST_NAME).
#define SPECTRUM_HIST_NAME "kc761_spectrum"

void subbkg(const std::string& sigFile, const std::string& bkgFile,
            const std::string& outFile = "") {
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

    TH1D* hSig = dynamic_cast<TH1D*>(fSig->Get(SPECTRUM_HIST_NAME));
    TH1D* hBkg = dynamic_cast<TH1D*>(fBkg->Get(SPECTRUM_HIST_NAME));
    TParameter<double>* tSig = dynamic_cast<TParameter<double>*>(fSig->Get("daq_time"));
    TParameter<double>* tBkg = dynamic_cast<TParameter<double>*>(fBkg->Get("daq_time"));

    if (!hSig || !hBkg || !tSig || !tBkg) {
        std::cerr << "[subbkg] error: files must contain TH1D \""
                  << SPECTRUM_HIST_NAME << "\" and "
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

    hSig->Sumw2();
    hBkg->Sumw2();

    TFile fOut(outName.c_str(), "RECREATE");
    fOut.cd();

    TH1D* hNet = new TH1D(*hSig);
    hNet->SetNameTitle(SPECTRUM_HIST_NAME,
                       "KC761 spectrum (background subtracted);Channel;Counts");
    hNet->Add(hSig, hBkg, 1.0, -r);

    hNet->Write();
    fOut.Close();

    std::cout << "[subbkg] wrote " << outName << " : " << nBins
              << " bins, scale r = t_sig/t_bkg = " << r << "\n";
}
