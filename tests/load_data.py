from urllib.request import urlopen
import numpy as np
import pandas as pd
import io

GITHUB_DATADIR = "https://raw.githubusercontent.com/Sm00thix/IKPLSTestData/main/data/"


def load_csv():
    csv_url = GITHUB_DATADIR + "dataset.csv"
    columns = [
        "Rye_Midsummer",
        "Wheat_H1",
        "Wheat_H3",
        "Wheat_H4",
        "Wheat_H5",
        "Wheat_Halland",
        "Wheat_Oland",
        "Wheat_Spelt",
        "Moisture",
        "Protein",
    ]
    csv = pd.read_csv(csv_url, usecols=columns)
    csv = csv.astype(np.float64)
    return csv


def load_spectra():
    spectra_url = GITHUB_DATADIR + "spectra.npz"
    resp = urlopen(spectra_url)
    resp_byte_array = resp.read()
    byte_contents = io.BytesIO(resp_byte_array)
    npz_arr = np.load(byte_contents)
    spectra = np.row_stack([npz_arr[k] for k in npz_arr.keys()])
    spectra = spectra.astype(np.float64)
    spectra = -np.log10(spectra)
    return spectra
