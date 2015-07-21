
import cPickle as pickle
import numpy as np
from glob import glob

def load_original_spectra(filename, **kwargs):
    return np.loadtxt(filename, skiprows=1, **kwargs)

def parse_star_name(filename):
    basename = filename.split("/")[-1]
    return basename.split("_")[0]

def create_data_arrays(filename_prefix, spectrum_filenames, wavelength_atol=1e-8, **kwargs):

    # Write star names to disk.
    star_names = map(parse_star_name, spectrum_filenames)
    with open("{}.pkl".format(filename_prefix), "w") as fp:
        pickle.dump(star_names, fp, -1) 
    
    # Get the number of pixels and a reference wavelength scale.
    lambda_reference = load_original_spectra(spectrum_filenames[0])[:, 0]
    num_pixels = lambda_reference.size
    num_spectra = len(spectrum_filenames)

    # The shape will contain the common wavelength scale in the first instance, the number of pixels,
    # and a two-dimensional array of flux, error values for each spectrum pixel
    data = np.memmap("{}.memmap".format(filename_prefix), dtype=float, mode="w+", shape=(num_spectra + 1, num_pixels, 2))

    # Save the lambda reference as the first 'spectrum'.
    data[0, :, 0] = lambda_reference
    data[0, :, 1] = 0 # Set the 'uncertainty' on lambda as zero.

    for i, spectrum_filename in enumerate(spectrum_filenames):
        if i < 230: continue
        spectrum_data = load_original_spectra(spectrum_filename, **kwargs)

        # Ensure the binning is the same.        
        np.allclose(spectrum_data[:, 0], lambda_reference, atol=wavelength_atol, rtol=1)
        data[i + 1, :, :] = spectrum_data[:, 1:]

        print("Loaded {0}/{1}: {2}".format(i, num_spectra, spectrum_filename))

    data.flush()
    del data
 
    return True



if __name__ == "__main__":
    

    # Write stuff to disk.
    filenames = glob("original-spectra/*.gz")
    
    # Ignore the Sun and 18Sco because they are on a different lambda scale
    n = len(filenames)
    filenames = list(set(filenames).difference(["original-spectra/18Sco_HARPS.txt.gz", "original-spectra/Sun_HARPS.txt.gz"])) 
    assert len(filenames) == n - 2
    create_data_arrays("hipparcos-spectra", filenames) 
