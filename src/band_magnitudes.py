import torch
import numpy as np
from enum import Enum

# This file contains multiple approaches to extracting individual band magnitudes
# If no specific band strengths for the cloud are provided, the cloud component
# has approximately evenly distributed cloud strength across channels, obtained
# using cloud_hue() or torch.ones_like()

# most methods are based on the ratio between the clear region of a real reference image and the cloudy region

def mean_mag(reference,mask,mask_cloudy=None,clean=None):    
    """ Extract ratios of means
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).mean()
        clear_val=(i[mask_clean]).mean() if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)

    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.mean() if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base    

    return cloud_mag

def max_mag(reference,mask,mask_cloudy=None,clean=None):
    """ Extract ratios of max values
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).max()
        clear_val=(i[mask_clean]).max() if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)

    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.median() if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base    

    return cloud_mag

def median_mag(reference,mask,mask_cloudy=None,clean=None):
    """ Extract ratios of medians
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).median()
        clear_val=(i[mask_clean]).median() if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)

    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.median() if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base    

    return cloud_mag

def q_mag(reference,mask,mask_cloudy=None, clean=None,q=0.95,q2=None):
    """ Extract ratios of quantiles
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
            q (float) : quantile value used for the cloudy region
            
            q2 (float) : optional quantile value used for the clear region (if unspecifed, it is equal to q)
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    if q2 is None:
        q2=q
    
    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).quantile(q)
        clear_val=(i[mask_clean]).quantile(q2) if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)
        
    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.quantile(q2) if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base

    return cloud_mag



class CloudType(Enum):
    """
    TODO: add more types

    Plan is to input reference mask with cloud types and adapt refl values accordingly
    """

    cloud = 1
    deep_connective_cloud = 2                                 
    thin_cloud = 3
    cirrus=4


"""
https://sentiwiki.copernicus.eu/web/s2-processing SCL algo
https://www.scribd.com/document/683252305/Cloud-Spectral-Reflectance (MODIS + IKNOS refl values of clouds)

(BELOW IS AI GEnerated and wrong!!!!)
Le Chat Mistral
Band,Wavelength (nm),Mean Reflectance,95% Confidence Interval (Lower–Upper)
B01,443,0.85,0.75–0.95
B02,490,0.90,0.80–1.00
B03,560,0.92,0.85–1.00
B04,665,0.90,0.80–1.00
B05,705,0.85,0.75–0.95
B06,740,0.85,0.75–0.95
B07,783,0.80,0.70–0.90
B08,842,0.90,0.80–1.00
B08A,865,0.85,0.75–0.95
B09,945,0.70,0.60–0.80
B10,1375,0.30,0.20–0.40 / thin cirrus detected from 0.012 to 0.035. 0.035 is the thick cloud threshold (for which sat?? (modis?))
B11,1610,0.20,0.10–0.30
B12,2190,0.10,0.05–0.15

Gemini
Top-of-Atmosphere (TOA) reflectance
Band,Name,Mean Reflectance,95% Low CI,95% Upper CI
B1,Coastal Aerosol,0.55,0.32,0.78
B2,Blue,0.58,0.35,0.81
B3,Green,0.59,0.36,0.82
B4,Red,0.60,0.37,0.83
B8,NIR (Broad),0.62,0.38,0.86
B8A,NIR (Narrow),0.63,0.39,0.87
B10,Cirrus,0.25*,0.05,0.65
B11,SWIR 1,0.35,0.15,0.55
B12,SWIR 2,0.22,0.08,0.36

Band,Name,Mean Reflectance,95% Low CI,95% Upper CI
B1,Coastal Aerosol,0.55,0.32,0.78
B2,Blue,0.58,0.35,0.81
B3,Green,0.59,0.36,0.82
B4,Red,0.60,0.37,0.83
B5,Red Edge 1,0.61,0.37,0.84
B6,Red Edge 2,0.61,0.38,0.85
B7,Red Edge 3,0.62,0.38,0.85
B8,NIR (Broad),0.62,0.38,0.86
B8A,NIR (Narrow),0.63,0.39,0.87
B10,Cirrus,0.25*,0.05,0.65
B11,SWIR 1,0.35,0.15,0.55
B12,SWIR 2,0.22,0.08,0.36

A feature? The Flatness Factor: For a cloud, the ratio of B7/B4 is usually close to 1.0.

Hollstein, A., et al. (2016). 
Ready-to-Use Methods for the Detection of Clouds, Cloud Shadows, Snow, and Water on Sentinel-2 Data. Remote Sensing.

Skakun, S., et al. (2022). Cloud Mask Intercomparison eXercise (CMIX): 
An evaluation of cloud masking algorithms for Landsat 8 and Sentinel-2. Remote Sensing of Environment.

ESA (2023/2024). 
Sentinel-2 MSI Annual Performance Report. Copernicus Sentinel-2 Mission Performance Centre.
"""

def stat_mag(reference_mask:torch.Tensor, mask, mask_cloudy=None, seed:int=42):
    """
        Use scientifically determined cloud spectral fingerprints (their mean and distribution function) in (sen2) bands
        to generate cloud magnitudes.

        Future:
            - differentiate cloud types and their spectral characteristica

    
        reference (Tensor) : input reference image containing cloud [height, width, channels]  
    
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    image_shape = reference_mask.shape

    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    shape = reference_mask.shape # (H,W,C)

    # if ever to change
    channel_map = {
        1:1,
        2:2,
        3:3,
        4:4,
        5:5,
        6:6,
        7:7,
        8:8,
        9:9,
        10:10, # cirrus, to be excluded TODO??
        11:11,
        12:12
    }

    # reflectance means and std deviations following
     # in order of bands
    channel_means = [0.55,0.58,0.59,0.60,0.61,0.61,0.62,0.62,0.63,0.25,0.35,0.22]
    channel_std_q_q2 = [
        [0.32,0.78],
        [0.35,0.81],
        [0.36,0.82],
        [0.38,0.85],
        [0.37,0.83],
        [0.37,0.84],
        [0.38,0.85],
        [0.38,0.86],
        [0.39,0.87],
        [0.05,0.65], # cirrus (0.012 - 0.035 thin cloud, > 0.035 thick cloud (with modis/ikonos/ at least))
        [0.15,0.55],
        [0.08,0.36], 
    ]
                                                          
    rng_band_coef_matrix = np.random.default_rng(seed=seed)\
        .normal(channel_means, channel_std_q_q2, size=(image_shape[0], image_shape[1]))

    rng_reflectance_matrix_clouds = mask_cloudy * rng_band_coef_matrix

    # values from 0. to 1. (magnitude) based on channel specific statistical values (magnitude)
    # shape (H, W, C)
    return rng_reflectance_matrix_clouds
