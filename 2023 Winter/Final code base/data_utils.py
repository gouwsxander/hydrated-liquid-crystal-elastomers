import numpy as np
import pandas as pd

DATA_ROOT = "2023 Winter/Final code base"

def get_data_frame(filename):
    df = pd.read_csv(f"{DATA_ROOT}/{filename}.csv")

    return df

def get_humidities(df):
    return df["RH (%)"]

def get_saturations(df, substance):
    """
    substance: either "collagen" or "wool"
    """
    humidities = get_humidities(df)

    df_saturation = get_data_frame("saturation_data")
    reference_humidities = np.array(df_saturation["RH (%)"])
    reference_saturations = np.array(df_saturation[f"g water per 100 g {substance}"])

    saturations = []
    for humidity in humidities:
        saturations.append(float(reference_saturations[reference_humidities == humidity]))

    return np.array(saturations)

def get_deformations(df, direction):
    """
    direction: either "axial" or "radial"
    """
    return df[f"{direction} deformation"], df[f"{direction} deformation uncertainty"]

def get_deformation_anisotropies(df):
    axial_deformations, axial_deformation_uncertainties = get_deformations(df, "axial")
    radial_deformations, radial_deformation_uncertainties = get_deformations(df, "radial")

    deformation_anisotropies = radial_deformations / axial_deformations

    radial_deformation_RSD = radial_deformation_uncertainties / radial_deformations
    axial_deformation_RSD = axial_deformation_uncertainties / axial_deformations
    deformation_anisotropy_uncertainties = deformation_anisotropies * np.sqrt((radial_deformation_RSD)**2 + (axial_deformation_RSD)**2)
    
    return deformation_anisotropies, deformation_anisotropy_uncertainties

def get_swell_ratios(df):
    axial_deformations, axial_deformation_uncertainties = get_deformations(df, "axial")
    radial_deformations, radial_deformation_uncertainties = get_deformations(df, "radial")

    swell_ratios = axial_deformations * radial_deformations**2

    radial_deformation_RSD = radial_deformation_uncertainties / radial_deformations
    axial_deformation_RSD = axial_deformation_uncertainties / axial_deformations
    swell_ratio_uncertainties = swell_ratios * np.sqrt((axial_deformation_RSD)**2 + 2 * (radial_deformation_RSD)**2)

    return swell_ratios, swell_ratio_uncertainties