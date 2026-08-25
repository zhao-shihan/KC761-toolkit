// csv2root.cxx
// ---------------------------------------------------------------------------
// Convert a KC761 multichannel-analyzer CSV export into a ROOT file.
//
// Input CSV format (as exported by the KC761 acquisition software):
//     Channel,Count #<D>d<H>h<M>m<S>s
//     0,0,
//     1,0,
//     ...
//   - Column 1: channel number (0 .. N-1)
//   - Column 2: counts per channel; its header contains the acquisition
//     (DAQ) time token "#0d22h41m31s" (days/hours/minutes/seconds).
//
// Output ROOT file contains:
//   - TH1D             "kc761_spectrum" : one bin per channel, bin center
//                                          == channel number
//   - TParameter<double> "daq_time"     : acquisition time in hours
//
// Usage:
//   root -l -b -q 'csv2root.cxx("bkg-260821-data.csv")'
//   root -l -b -q 'csv2root.cxx("bkg-260821-data.csv","out.root")'
// ---------------------------------------------------------------------------

#include "TFile.h"
#include "TH1D.h"
#include "TParameter.h"
#include "TSystem.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Parse an acquisition-time token of the form "#0d22h41m31s"
// (days / hours / minutes / seconds) and return it in hours.
double ParseDaqTimeHours(const std::string& header)
{
    size_t pos = header.find('#');
    if (pos == std::string::npos) {
        std::cerr << "[csv2root] warning: no '#<D>d<H>h<M>m<S>s' acquisition-time "
                  << "token found in header: \"" << header << "\"\n";
        return 0.0;
    }

    double days = 0.0, hours = 0.0, minutes = 0.0, seconds = 0.0;
    std::string num;
    for (size_t i = pos + 1; i < header.size(); ++i) {
        char c = header[i];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            num.push_back(c);
        } else if (c == 'd') {
            days = std::atof(num.c_str()); num.clear();
        } else if (c == 'h') {
            hours = std::atof(num.c_str()); num.clear();
        } else if (c == 'm') {
            minutes = std::atof(num.c_str()); num.clear();
        } else if (c == 's') {
            seconds = std::atof(num.c_str()); num.clear();
            break;
        } else {
            break; // unexpected char -> stop parsing
        }
    }
    return days * 24.0 + hours + minutes / 60.0 + seconds / 3600.0;
}

} // namespace

void csv2root(const std::string& input, const std::string& output = "")
{
    // Default output name: input filename with extension replaced by ".root".
    std::string outName = output;
    if (outName.empty()) {
        outName = input;
        size_t dot = outName.rfind('.');
        if (dot != std::string::npos) outName = outName.substr(0, dot);
        outName += ".root";
    }

    std::ifstream in(input.c_str());
    if (!in.is_open()) {
        std::cerr << "[csv2root] error: cannot open input file: " << input << "\n";
        gSystem->Exit(1);
        return;
    }

    // Column-2 header carries the acquisition time.
    std::string headerLine;
    std::getline(in, headerLine);
    double daqHours = ParseDaqTimeHours(headerLine);

    std::vector<int> channels;
    std::vector<double> counts;

    std::string line;
    int lineNo = 1; // header already consumed
    while (std::getline(in, line)) {
        ++lineNo;
        size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue; // blank line

        int ch = -1;
        double cnt = 0.0;
        char comma = 0;
        std::istringstream ss(line.substr(start));
        ss >> ch >> comma >> cnt;
        if (comma != ',' || ch < 0 || ss.fail()) {
            std::cerr << "[csv2root] warning: skipping malformed line " << lineNo
                      << ": \"" << line << "\"\n";
            continue;
        }
        channels.push_back(ch);
        counts.push_back(cnt);
    }

    if (channels.empty()) {
        std::cerr << "[csv2root] error: no data rows parsed from " << input << "\n";
        gSystem->Exit(1);
        return;
    }

    int nCh = 0;
    for (int ch : channels)
        if (ch + 1 > nCh) nCh = ch + 1;

    // Bin center == channel number  =>  range [-0.5, nCh - 0.5].
    TH1D* h = new TH1D("kc761_spectrum", "kc761 spectrum;Channel;Counts",
                       nCh, -0.5, nCh - 0.5);
    for (size_t i = 0; i < counts.size(); ++i) {
        int bin = channels[i] + 1;
        if (bin >= 1 && bin <= nCh) h->SetBinContent(bin, counts[i]);
    }

    TFile f(outName.c_str(), "RECREATE");
    h->Write();
    TParameter<double>* daqTime = new TParameter<double>("daq_time", daqHours);
    daqTime->Write();
    f.Close();

    std::cout << "[csv2root] wrote " << outName << " : " << nCh
              << " channels, integral " << h->Integral() << ", daq_time = "
              << daqHours << " h\n";
}
