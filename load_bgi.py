"""
This module is used to read BGI data and image file, and return an AnnData object.
"""

import itertools
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from skimage import io
from scipy.sparse import csr_matrix # type: ignore
from anndata import AnnData

import cv2


def load_bin(
    gem_file: str,
    image_file: str,
    bin_size: int,
    library_id: str,
) -> AnnData:
    """
    Read BGI data and image file, and return an AnnData object.
    Parameters
    ----------
    gem_file
        The path of the BGI data file.
    image_file
        The path of the image file.
    bin_size
        The size of the bin.
    library_id
        The library id.
    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
    """ # noqa: E501
    library = library_id
    dat_file = gem_file
    image = image_file
    bin_s = bin_size
    ###########################
    # different gem have different delimiter!!!!!!!
    # COAD: " " , other may be "\t"
    dat = pd.read_csv(dat_file, delimiter="\t", comment="#")
    
    image = cv2.imread(image)
    ######
    dat['x'] -= dat['x'].min()
    dat['y'] -= dat['y'].min()

    width = dat['x'].max() + 1
    height = dat['y'].max() + 1
    ###
    dat['xp'] = (dat['x'] // bin_s) * bin_s
    dat['yp'] = (dat['y'] // bin_s) * bin_s
    dat['xb'] = np.floor(dat['xp'] / bin_s + 1).astype(int)
    dat['yb'] = np.floor(dat['yp'] / bin_s + 1).astype(int)

    dat['bin_ID'] = max(dat['xb']) * (dat['yb'] - 1) + dat['xb']
    ###
    trans_x_xb = dat[['x', 'xb']].drop_duplicates()
    trans_x_xb = trans_x_xb.groupby('xb')['x'].apply(
        lambda x: int(np.floor(np.mean(x)))).reset_index()
    trans_y_yb = dat[['y', 'yb']].drop_duplicates()
    trans_y_yb = trans_y_yb.groupby('yb')['y'].apply(
        lambda y: int(np.floor(np.mean(y)))).reset_index()

    trans_matrix = pd.DataFrame(list(itertools.product(
        trans_x_xb['xb'], trans_y_yb['yb'])), columns=['xb', 'yb'])
    trans_matrix = pd.merge(trans_matrix, trans_x_xb, on='xb')
    trans_matrix = pd.merge(trans_matrix, trans_y_yb, on='yb')
    trans_matrix['bin_ID'] = max(
        trans_matrix['xb']) * (trans_matrix['yb'] - 1) + trans_matrix['xb']

    trans_matrix['in_tissue'] = 1

    tissue_positions = pd.DataFrame()
    # barcode is str, not number
    tissue_positions['barcodes'] = trans_matrix['bin_ID'].astype(str)
    tissue_positions['in_tissue'] = trans_matrix['in_tissue']
    tissue_positions['array_row'] = trans_matrix['yb']
    tissue_positions['array_col'] = trans_matrix['xb']
    tissue_positions['pxl_row_in_fullres'] = trans_matrix['y']
    tissue_positions['pxl_col_in_fullres'] = trans_matrix['x']
    tissue_positions.set_index('barcodes', inplace=True)

    ### 
    if 'MIDCount' in dat.columns:
        dat = dat.groupby(['geneID', 'xb', 'yb'])[
            'MIDCount'].sum().reset_index()
        dat['bin_ID'] = max(dat['xb']) * (dat['yb'] - 1) + dat['xb']

        ### 
        unique_genes = dat['geneID'].unique()
        unique_barcodes = dat['bin_ID'].unique()
        gene_hash = {gene: index for index, gene in enumerate(unique_genes)}
        barcodes_hash = {barcodes: index for index,
                         barcodes in enumerate(unique_barcodes)}
        dat['gene'] = dat['geneID'].map(gene_hash)
        dat['barcodes'] = dat['bin_ID'].map(barcodes_hash)

        ### 
        counts = csr_matrix((dat['MIDCount'], (dat['barcodes'], dat['gene'])))

    else:
        dat = dat.groupby(['geneID', 'xb', 'yb'])[
            'MIDCounts'].sum().reset_index()
        dat['bin_ID'] = max(dat['xb']) * (dat['yb'] - 1) + dat['xb']
        ###
        unique_genes = dat['geneID'].unique()
        unique_barcodes = dat['bin_ID'].unique()
        gene_hash = {gene: index for index, gene in enumerate(unique_genes)}
        barcodes_hash = {barcodes: index for index,
                         barcodes in enumerate(unique_barcodes)}
        dat['gene'] = dat['geneID'].map(gene_hash)
        dat['barcodes'] = dat['bin_ID'].map(barcodes_hash)

        ###
        counts = csr_matrix((dat['MIDCounts'], (dat['barcodes'], dat['gene'])))
    adata = AnnData(counts)
    adata.var_names = list(gene_hash.keys())
    adata.obs_names = list(map(str, barcodes_hash.keys()))
    ##########
    adata.obs = adata.obs.join(tissue_positions, how="left")
    adata.obsm['spatial'] = adata.obs[[
        'pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
    adata.obs.drop(columns=['in_tissue', 'array_row', 'array_col',
                   'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True,)
    ###
    spatial_key = "spatial"
    adata.uns[spatial_key] = {library: {}}
    adata.uns[spatial_key][library]["images"] = {}
    adata.uns[spatial_key][library]["images"] = {"hires": image}
    # tissue image / RNA shape
    tissue_hires_scalef = max(image.shape[0]/width, image.shape[1]/height)

    # the diameter of detection area(the spot that contains tissue)
    # can be adjust out side by size= in scatter function
    spot_diameter = bin_s / tissue_hires_scalef
    
    #fiducial_area = max(tissue_positions['array_row'].max() - tissue_positions['array_row'].min(),
    #                    tissue_positions['array_col'].max() - tissue_positions['array_col'].min())
    adata.uns[spatial_key][library]["scalefactors"] = {
        "tissue_hires_scalef": tissue_hires_scalef,
        "spot_diameter_fullres": spot_diameter,
    }

    return adata


def load_cell(
    gem_file: str,
    image_file: str,
    mask_file: int,
    library_id: str,
) -> AnnData:
    """
    Read BGI data and image file, and return an AnnData object.
    Parameters
    ----------
    gem_file
        The path of the BGI data file.
    image_file
        The path of the image file.
    bin_size
        The size of the bin.
    library_id
        The library id.
    Returns
    -------
    Annotated data object with the following keys:

        - :attr:`anndata.AnnData.obsm` ``['spatial']`` - spatial spot coordinates.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['images']`` - *hires* images.
        - :attr:`anndata.AnnData.uns` ``['spatial']['{library_id}']['scalefactors']`` - scale factors for the spots.
    """ # noqa: E501
    
    # mask = pd.read_csv(mask_file, delimiter=",")
    dat = pd.read_csv(gem_file, delimiter="\t", comment="#")
    image = io.imread(image_file)
    
    spatial_key = 'spatial'
    library = library_id

    mask = np.load(mask_file)
    mask_nozero = np.nonzero(mask)
    x = mask_nozero[0]; y = mask_nozero[1]
    value = [mask[x[i],y[i]] for i in range(len(x))]
    mask = pd.DataFrame({'x':y,'y':x,'barcodes':value})
    
    # stereoseq GEM xy is not the same as image xy
    # exchange gem xy!
    # !!!!!!!!!!!!!!!! 
    ##############y shall we? yes!
    # dat = dat.rename(columns={'x': 'temp'})
    # dat = dat.rename(columns={'y': 'x'})
    # dat = dat.rename(columns={'temp': 'y'})  
    # ######### 
    dat['x'] -= dat['x'].min()
    dat['y'] -= dat['y'].min()
    mask['x'] = mask['x'] - mask['x'].min()
    mask['y'] = mask['y'] - mask['y'].min() 
    # 20230717
    # dat['y'] = dat['y'].max() - dat['y'] # 为什么错了呢？
    # dat['y'] = dat['y'][::-1]
    
    mask_data = pd.merge(left = mask, right = dat, on=['x', 'y'],how = "inner")

    # mask_data for RNA
    # mask for celluar location
    exp = mask_data.groupby(['geneID', 'barcodes'])['MIDCount'].sum().reset_index()
    
    # construct count matrix
    unique_genes = exp['geneID'].unique()
    unique_barcodes = exp['barcodes'].unique()
    gene_hash = {gene: index for index, gene in enumerate(unique_genes)}
    barcodes_hash = {barcodes: index for index, barcodes in enumerate(unique_barcodes)}

    exp['gene'] = exp['geneID'].map(gene_hash)
    exp['barcodes'] = exp['barcodes'].map(barcodes_hash)
 
    counts = csr_matrix((exp['MIDCount'], (exp['barcodes'], exp['gene']))) 

    adata = AnnData(counts)
    adata.var_names = list(gene_hash.keys())
    adata.obs_names = list(map(str, barcodes_hash.keys()))

    # normalize mask coordinate to get mask and data overlap region
    # this is to ensure cell position start from left upper corner
    # according to mask, for we only care about 

    grouped_mask = mask.groupby('barcodes')
    transform_mtx = pd.DataFrame(columns=['barcodes', 'center_x', 'center_y'])

    for barcode, group in grouped_mask:
        x_mean = int(np.floor(group['x'].mean()))
        y_mean = int(np.floor(group['y'].mean()))
        
        # yanping 2023-07-03
        # 'center_x': x_mean, 'center_y': y_mean
        #transform_mtx = transform_mtx.append({'barcodes': str(barcode), 'center_x': x_mean, 'center_y': y_mean}, ignore_index=True)
        transform_mtx = transform_mtx._append({'barcodes': str(barcode), 'center_x': x_mean, 'center_y': y_mean}, ignore_index=True)
    
    # reset index
    transform_mtx.set_index('barcodes', inplace=True)

    adata.obs = adata.obs.join(transform_mtx, how="left")
    adata.obsm['spatial'] = adata.obs[[ "center_y","center_x"]].to_numpy()
    
    adata.obs.drop(columns=["center_x", "center_y"], inplace=True)
    
    adata.uns[spatial_key] = {library: {}}
    adata.uns[spatial_key][library]["images"] = {"hires": image}
    ######
    
    tissue_hires_scalef = max((mask['y'].max()+1)/image.shape[1], (mask['x'].max()+1)/image.shape[0])
    
    # spot_diameter could be set to *mean pixel of mask* / hires_scalef
    # can be adjust out side by size= in scatter function
    adata.uns[spatial_key][library]["scalefactors"] = {
         "tissue_hires_scalef": tissue_hires_scalef,
         "spot_diameter_fullres": 250,
    }
    
    return adata