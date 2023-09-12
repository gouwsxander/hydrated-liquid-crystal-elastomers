import numpy as np
import pandas as pd

#DATA_ROOT = "2023 Winter/Final code base" # data root for windows
DATA_ROOT = "./Documents/GitHub/hydrated-liquid-crystal-elastomers/2023 Winter/Final code base"

def get_data_frame(filename):
    df = pd.read_csv(f"{DATA_ROOT}/{filename}.csv")

    return df

def get_humidities(df):
    return df["RH (%)"]

def get_saturations(df, substance):
    """
    substance: either "collagen" or "wool"
    """
    humidities = np.array(get_humidities(df))

    df_saturation = get_data_frame(f"saturation_data_{substance}")
    reference_humidities = np.array(df_saturation["RH (%)"])
    reference_saturations = np.array(df_saturation[f"g water per 100 g {substance}"])

    saturations = []
    for humidity in humidities:
        for reference_index in range(len(reference_humidities) - 1):
            humidity_1 = reference_humidities[reference_index]
            humidity_2 = reference_humidities[reference_index + 1]
            if humidity_1 <= humidity and humidity <= humidity_2:
                saturation_1 = reference_saturations[reference_index]
                saturation_2 = reference_saturations[reference_index + 1]

                interpolated_saturation = (1 - (humidity - humidity_1) / (humidity_2 - humidity_1)) * saturation_1 + ((humidity - humidity_1) / (humidity_2 - humidity_1)) * saturation_2
                saturations.append(interpolated_saturation)
                break

        #try:
        #    saturations.append(float(reference_saturations[reference_humidities == humidity]))
        #except:
        #    print(f"Warning: Humidity {humidity} not seen.")

    return np.array(saturations) / 100 # return g/g

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

def get_strain_anisotropies(df):
    axial_deformations, axial_deformation_uncertainties = get_deformations(df, "axial")
    radial_deformations, radial_deformation_uncertainties = get_deformations(df, "radial")

    strain_anisotropies = (radial_deformations - 1) / (axial_deformations - 1)

    numerator_RSD = radial_deformation_uncertainties / (radial_deformations - 1)
    denominator_RSD = axial_deformation_uncertainties / (axial_deformations - 1)
    strain_anisotropy_uncertainties = strain_anisotropies * np.sqrt(numerator_RSD**2 + denominator_RSD**2)
    
    return strain_anisotropies, strain_anisotropy_uncertainties