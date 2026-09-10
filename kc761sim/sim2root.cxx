/* sim2root.cxx
   Convert a kc761sim simulation-file export (written by kc761sim.export)
   into a ROOT file:

     * primary_to_deposition  TH2D, x = energy deposition, y = primary
       energy (both axes non-uniform: the input calibration file's
       deposition-energy edges), fSumw2 = the per-bin variance (the
       Gaussian approximation var ~ counts, unit-weight fills);
     * detection_efficiency   TH1D per primary column (1 - zero-deposition
       fraction);
     * metadata TParameters: mode, mode_name, geometry_name, angular,
       geometry_param, seed, n_events, calib_binning_source (+ sha256).

   The primary-to-channel composition R = C @ G happens at unfold time
   (kc761unfold), which receives both the calibration file and this
   simulation file.
*/

#include "../kc761util/rootmacros.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const char* kMagic = "kc761sim-sim-export-v1";

static std::FILE* gBin = nullptr;

static void readBlock(std::vector<double>& buf, int id, const char* what) {
    std::int64_t rid = 0;
    if (std::fread(&rid, sizeof(rid), 1, gBin) != 1) {
        Fatal(std::string("unexpected end of file reading block ") + what);
        return;
    }
    if (rid != id) {
        Fatal(std::string("block id mismatch for ") + what + " (got "
              + std::to_string(rid) + ", expected " + std::to_string(id)
              + ")");
        return;
    }
    if (std::fread(buf.data(), sizeof(double), buf.size(), gBin)
        != (std::size_t)buf.size()) {
        Fatal(std::string("unexpected end of file reading block ") + what);
        return;
    }
}

void sim2root(const char* exportPath, const char* outputPath) {
    /* --- text header: magic + five '\n'-terminated lines --- */
    gBin = std::fopen(exportPath, "rb");
    if (!gBin) {
        Fatal(std::string("cannot open ") + exportPath);
        return;
    }
    std::string line;
    std::string modeName, geometryName, angular, calibPath, calibSha;
    if (!ReadLine(gBin, line)) {
        Fatal("cannot read the magic line");
        return;
    }
    StripNewline(line);
    if (line != kMagic) {
        Fatal("not a kc761sim simulation-file export (bad magic)");
        return;
    }
    if (!ReadLine(gBin, modeName) || !ReadLine(gBin, geometryName) ||
        !ReadLine(gBin, angular) || !ReadLine(gBin, calibPath) ||
        !ReadLine(gBin, calibSha)) {
        Fatal("truncated text header");
        return;
    }
    StripNewline(modeName);
    StripNewline(geometryName);
    StripNewline(angular);
    StripNewline(calibPath);
    StripNewline(calibSha);

    /* --- binary section --- */
    std::int64_t n = 0, mode = 0;
    if (std::fread(&n, sizeof(n), 1, gBin) != 1 ||
        std::fread(&mode, sizeof(mode), 1, gBin) != 1) {
        Fatal("cannot read the header integers");
        return;
    }
    const int nCh = (int)n;
    const std::size_t nEntries = (std::size_t)nCh * (std::size_t)nCh;
    if (nCh < 1 || nCh > (1 << 14)) {
        Fatal("unsupported matrix size " + std::to_string(nCh));
        return;
    }

    std::vector<double> edges(nCh + 1);
    double geometryParam = 0.0, nEventsD = 0.0, seedD = 0.0;
    if (std::fread(edges.data(), sizeof(double), edges.size(), gBin)
            != edges.size() ||
        std::fread(&geometryParam, sizeof(double), 1, gBin) != 1 ||
        std::fread(&nEventsD, sizeof(double), 1, gBin) != 1 ||
        std::fread(&seedD, sizeof(double), 1, gBin) != 1) {
        Fatal("cannot read the header arrays");
        return;
    }

    std::vector<double> counts(nEntries);
    readBlock(counts, 1, "primary-to-deposition counts");
    std::vector<double> sumw2(nEntries);
    readBlock(sumw2, 2, "primary-to-deposition sumw2");
    std::vector<double> efficiency(nCh);
    readBlock(efficiency, 3, "detection efficiency");
    std::fclose(gBin);
    gBin = nullptr;

    /* --- ROOT output --- */
    TFile out(outputPath, "RECREATE");
    if (out.IsZombie()) {
        Fatal(std::string("cannot create ") + outputPath);
        return;
    }

    TH2D* hG = new TH2D("primary_to_deposition",
                        "KC761 primary-to-deposition matrix;"
                        "Energy deposition (keV);"
                        "Primary gamma energy (keV)",
                        nCh, edges.data(), nCh, edges.data());
    FillMatrix(hG, counts, sumw2, nCh, "primary_to_deposition");
    hG->Write();

    TH1D* hEff = new TH1D("detection_efficiency",
                          "detection efficiency;primary energy (keV);eff",
                          nCh, edges.data());
    for (int j = 0; j < nCh; ++j) {
        hEff->SetBinContent(j + 1, efficiency[j]);
    }
    hEff->Write();

    TParameter<double> pMode("mode", (double)mode);
    pMode.Write();
    TNamed pModeName("mode_name", modeName.c_str());
    pModeName.Write();
    TNamed pGeomName("geometry_name", geometryName.c_str());
    pGeomName.Write();
    TNamed pAngular("angular", angular.c_str());
    pAngular.Write();
    TParameter<double> pGeomParam("geometry_param", geometryParam);
    pGeomParam.Write();
    TParameter<double> pSeed("seed", seedD);
    pSeed.Write();
    TParameter<double> pEvents("n_events", nEventsD);
    pEvents.Write();
    TNamed pCalibSrc("calib_binning_source", calibPath.c_str());
    pCalibSrc.Write();
    TNamed pCalibSha("calib_binning_sha256", calibSha.c_str());
    pCalibSha.Write();

    out.Close();
    std::remove(exportPath);
    std::cout << "[sim2root] wrote " << outputPath << " : "
              << nCh << " x " << nCh
              << " primary-to-deposition matrix with per-bin variances, "
                 "detection efficiency and run metadata\n";
}
