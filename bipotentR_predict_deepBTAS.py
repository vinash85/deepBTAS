"""Evaluates the model"""

import argparse
import logging
import os


from scipy.stats import norm, rankdata
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
# import model.net as net
# import model.data_generator as data_generator

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='input.txt', help="Input file with gene expression (Sample x Genes). Must contain header with ensembl gene names. All genes input are required!")
parser.add_argument('--output_file', default='output.txt', help="Output file with prediction (Sample x 1).")
parser.add_argument('--saved_model_dir', default='best', help="name of the director in in which embedding_model (embedding_model.pth and output_model.pth are stored. Make sure the saved models name are exact")
parser.add_argument('--type_prediction', default='immune_metabolism',
                    help=" ('immune_metabolism or angiogenesis) Prediction based on immune metabolism bipotent targets  \n \
                    or angiogenesis + growth - suprressor bipotent targets")
match = lambda a, b: [b.index(x) if x in b else None for x in a]


def is_binary(a):
    return ((a == 0) | (a == 1)).all()


def is_categorical(xx):
    return any([isinstance(uu, str) for uu in xx])


def np_take(aa, indices, axis=0):
    try:
        out = np.take(aa, indices, axis)
    except:
        shape1 = aa.shape
        new_shape = list(shape1)
        new_shape[axis] = len(indices)
        indices = np.array(indices)
        inx = np.where(indices < shape1[axis])
        indices_nan = indices[inx]
        out = np.empty(new_shape)
        out[:] = np.nan
        out = out.astype(aa.dtype)
        for ii, ind in enumerate(indices_nan):
            if(len(shape1) == 1):
                out[ii] = aa[ind]
            else:
                out[:, ii] = aa[:, ind]

    return out


def readFile(input_file, header=False):
    data = pd.read_csv(input_file, sep="\t")
    out = data.values
    if header:
        header = list(data.columns)
        out = (out, header)
    return out


def qnorm_array(xx):
    """
    Perform quantile normalization on a np.array similar to qnorm_col
    """
    xx_back = xx
    xx = xx_back[~np.isnan(xx)]
    if len(xx) > 0:
        if np.nanstd(xx) > 0:
            xx = rankdata(xx, method="average")
            xx = xx / (max(xx) + 1)  # E(max(x)) < sqrt(2log(n))
            xx = norm.ppf(xx, loc=0, scale=1)
            xx_back[~np.isnan(xx_back)] = xx
        else:
            xx[:] = 0

    return xx_back


def quantile_normalize_nonbinary(data, method="qnorm_array"):
    """
    Perform quantile normalization on a np.array similar to qnorm_col  (only to nonbinary data)
    """

    # force data into floats for np calculations
    data = data.astype('float')
    method = eval(method)
    for index in range(data.shape[1]):
        col = data[:, index]
        if not is_binary(np.nan_to_num(col)):
            data[:, index] = method(col)

    return data


def process_inputs(features):
    """
    Parses the input file and creates a generator for the input file

    Returns:
    data_generator() -- the generator function to yield the features and labels
    """
    # np.random.seed(230)
    # tracer()

    features = features.astype(float)
    features = np.nan_to_num(features)  # convert NANs to zeros
    features = quantile_normalize_nonbinary(features)
    return features


if __name__ == '__main__':
    args = parser.parse_args()
    output_file = args.output_file
    input_file = args.input_file
    saved_model_dir = args.saved_model_dir
    embedding_path = saved_model_dir + "/embedding_model_scripted.pt"
    output_model_path = saved_model_dir + "/outputs_scripted.pt"
# output_file = "/homes6/asahu/temp.csv"
# embedding_path = "/homes6/asahu/project/deeplearning/icb/data/ESRRA/saved.models/ESRRA.38.mixed.training.mixed_20220525-233307/embedding_model.pth"
# output_model_path = "/homes6/asahu/project/deeplearning/icb/data/ESRRA/saved.models/ESRRA.38.mixed.training.mixed_20220525-233307/output_model.pth"
# input_file = "/homes6/asahu/project/deeplearning/icb/data/ESRRA/TRIM.38.icb/dataset.txt"
    immune_metabolism_ensembl_input_genes = ['ENSG00000072364',
                                             'ENSG00000189079',
                                             'ENSG00000204256',
                                             'ENSG00000134058',
                                             'ENSG00000118260',
                                             'ENSG00000101412',
                                             'ENSG00000169016',
                                             'ENSG00000120690',
                                             'ENSG00000151702',
                                             'ENSG00000101216',
                                             'ENSG00000125651',
                                             'ENSG00000164683',
                                             'ENSG00000117139',
                                             'ENSG00000103495',
                                             'ENSG00000025434',
                                             'ENSG00000185551',
                                             'ENSG00000123358',
                                             'ENSG00000140464',
                                             'ENSG00000117222',
                                             'ENSG00000111424']

    angiogenesis_ensembl_input_genes = ["ENSG00000156127", "ENSG00000082258", "ENSG00000164330", "ENSG00000160973", "ENSG00000174332", "ENSG00000171988", "ENSG00000103495", "ENSG00000197157", "ENSG00000269404", "ENSG00000073861", "ENSG00000187079", "ENSG00000198176", "ENSG00000073282", "ENSG00000125482"]

    im = "immune_metabolism"
    if args.type_prediction == im:
        ensembl_input_genes = immune_metabolism_ensembl_input_genes
    else:
        ensembl_input_genes = angiogenesis_ensembl_input_genes

    # embedding_model = torch.load(embedding_path)
    # outputs = torch.load(output_model_path)
    embedding_model = torch.jit.load(embedding_path)
    outputs = torch.jit.load(output_model_path)

    embedding_model = embedding_model.cpu()
    outputs = outputs.cpu()
    embedding_model.eval()
    outputs.eval()

# path = "/homes6/asahu/temp.pth"

    input_matrix, header = readFile(input_file, header=True)

    all_genes_present = all(elem in header for elem in ensembl_input_genes)
    if not all_genes_present:
        print("Some input genes required are absent in the input file")

        ensembl_input_genes_str = '\n'.join(map(str, ensembl_input_genes))
        print("Genes input required: \n" + ensembl_input_genes_str)
        sys.exit(1)

    match_inx = match(ensembl_input_genes, header)

    input_matrix1 = np_take(input_matrix, match_inx, axis=1)
    features = process_inputs(input_matrix1)

    data_batch = torch.from_numpy(features).float()
    embedding_batch = embedding_model(data_batch)

    output_batch = outputs(embedding_batch)

    predictions = output_batch.data.numpy()

    predictions_df = pd.DataFrame(predictions)

    predictions_df.columns = ["prediction"]
    predictions_df.to_csv(output_file, sep='\t', index=False)
