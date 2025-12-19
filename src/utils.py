# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 24.09.2024

import glob
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import matplotlib
import nibabel as nib
import requests
import SimpleITK as sitk
from colorama import Fore, Style, init

matplotlib.use("Agg")  # Use non-interactive backend
import numpy as np
from matplotlib import pyplot as plt

# import warnings
# Initialize colorama for cross-platform support
init(autoreset=True)
try:
    columns = os.get_terminal_size().columns
except:
    columns = 80


def print_completed(mypath):
    print(Fore.GREEN + Style.BRIGHT + f"Finished: {mypath}")


def print_ensemble_message():
    # Aesthetic header
    citation_title = "If you are using The Isles'22 Ensemble algorithm, please cite the following work:"
    citation_text = (
        "de la Rosa, E. et al. (2024) A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: "
        "Generalizability and Clinical Utility Beyond the ISLES Challenge. arXiv:2403.19425."
    )

    # Define the maximum width for each line in the terminal
    max_width = 120

    # Wrap the citation text
    wrapped_citation = textwrap.fill(citation_text, max_width)

    # Print the header with formatting
    print(Fore.WHITE + "#" * (max_width + 4))
    print(Fore.WHITE + "#" * (max_width + 4))
    print(Fore.BLUE + citation_title)

    # Print the citation with line breaks
    print(Fore.YELLOW + Style.BRIGHT + wrapped_citation)

    # Print the footer with formatting
    print(Fore.WHITE + "#" * (max_width + 4))
    print(Fore.WHITE + "#" * (max_width + 4))


def print_run(algorithm):
    print("Running {} algorithm ...".format(algorithm))


def get_img_shape(image_path):
    myimg = nib.load(image_path)
    return len(myimg.shape)


def save_nii(mydata, myaffine, myheader, outpath):
    nib.save(nib.Nifti1Image(mydata, myaffine, myheader), outpath)


def convert_to_nii(input_path, tmp_dir, image_mod):
    # case dcm
    if Path(input_path).is_dir():  # dcm folder
        if any(Path(input_path).rglob("*.dcm")):
            output_dir = os.path.join(tmp_dir, image_mod)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            command = [
                "dcm2niix",  # The dcm2niix command
                "-z",
                "y",  # Compress the output NIfTI files (.nii.gz)
                "-o",
                output_dir,  # Output directory
                input_path,  # Directory containing DICOM files
            ]

            print("Converting {} dicom to nifti...".format(image_mod))
            with open(os.devnull, "w") as devnull:
                subprocess.run(command, stdout=devnull, stderr=devnull, check=True)

            new_path = os.path.join(output_dir, "{}.nii.gz".format(image_mod))
            os.rename(glob.glob(os.path.join(output_dir, "*.nii.gz"))[0], new_path)

            return new_path, False  # flag to indicate dcm/nii

    # case .nii
    else:
        if (
            input_path[-4:] == ".nii"
            or input_path[-7:] == ".nii.gz"
            or input_path[-4:] == ".mha"
        ):
            output_dir = os.path.join(tmp_dir, image_mod)
            output_dir_file = os.path.join(output_dir, image_mod + ".nii.gz")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            shutil.copyfile(input_path, output_dir_file)
            return output_dir_file, True  # flag to indicate dcm/nii
        else:
            raise ValueError("No .nii, .nii.gz, .mha, or Dicom files available.")


def extract_brain(input_path, output_path, gpu=True, save_mask=0):
    if gpu:
        command_hd_bet = (
            f"hd-bet -i {input_path} -o {output_path} -s {save_mask} -mode fast"
        )
    else:
        command_hd_bet = f"hd-bet -i {input_path} -o {output_path} -s {save_mask} -device cpu -mode fast -tta 0"

    # Run HD-BET while suppressing warnings (stderr) but keeping print output (stdout)
    with open(os.devnull, "w") as devnull:
        subprocess.call(command_hd_bet, shell=True, stderr=devnull)


def check_gpu_memory(min_free_memory_gb=12):
    try:
        # Run the `nvidia-smi` command to get GPU memory details
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        # Parse the output
        free_memory_list = result.stdout.strip().split("\n")
        free_memory_list = [
            int(mem) for mem in free_memory_list
        ]  # Convert memory values to integers (in MB)

        # Check if any GPU has sufficient free memory
        for free_memory_mb in free_memory_list:
            free_memory_gb = free_memory_mb / 1024  # Convert MB to GB
            # print(free_memory_gb)
            if free_memory_gb >= min_free_memory_gb:
                return True

        return False

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Error occurred while trying to check GPU memory: {e.stderr}")
        else:
            # FileNotFoundError when nvidia-smi is not installed
            pass
        return False


#    shutil.


def register_mri(
    fixed_image_path, moving_image_path, out_dir_path, transformation="rigid"
):
    # Set up the ElastixImageFilter
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(moving_image)
    elastix.SetOutputDirectory(os.path.dirname(out_dir_path))

    elastix.SetParameterMap(sitk.GetDefaultParameterMap(transformation))
    elastix.LogToConsoleOff()
    elastix.LogToFileOff()
    # elastix.AddParameterMap(sitk.GetDefaultParameterMap("affine"))

    elastix.Execute()

    reg_image = elastix.GetResultImage()

    # Optionally, save the registered image
    sitk.WriteImage(reg_image, out_dir_path)
    # Execute the registration


def propagate_image(mask_image_path, out_dir_path, is_mask=False):
    mask_image = sitk.ReadImage(mask_image_path)
    transform_param_files = glob.glob(
        os.path.join(os.path.dirname(out_dir_path), "TransformParameters.*.txt")
    )

    for param_file in transform_param_files:
        # Read the parameter map and set nearest neighbor interpolator for binary masks
        transform_param_map = sitk.ReadParameterFile(param_file)
        if is_mask:
            transform_param_map["ResampleInterpolator"] = [
                "FinalNearestNeighborInterpolator"
            ]

        # Apply the transformation using Transformix
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(mask_image)
        transformix.SetTransformParameterMap(transform_param_map)
        transformix.LogToConsoleOff()
        transformix.LogToFileOff()
        # Execute the transformation
        transformix.Execute()

        # Get the transformed mask image for the next iteration
        mask_image = transformix.GetResultImage()
    # Save the final transformed binary mask
    sitk.WriteImage(mask_image, out_dir_path)


def get_flair_atlas(output_path):
    """Get a FLAIR-MNI vascular territory atlas from https://zenodo.org/records/3379848
    https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2019.00208/full
    """

    url_flair = url = (
        "https://zenodo.org/record/3379848/files/caa_flair_in_mni_template_smooth_brain_intres.nii.gz?download=1"
    )

    # Download the file if it does not exist
    if not os.path.exists(output_path):
        print("Getting vascular territory atlas. If you use it, please cite:")
        print(
            ' Schirmer, Markus D., et al. "Spatial signature of white matter hyperintensities in stroke patients." Frontiers in neurology 10 (2019): 208.'
        )

        response = requests.get(url_flair)
        with open(output_path, "wb") as f:
            f.write(response.content)


# Load brain mask if provided
def _load_slice_from_nifti(nifti_path, slice_idx, axis="axial", nii_img=None):
    """
    Load a specific slice from a NIfTI file.
    Memory-efficient: if nii_img is provided, reuses it to avoid reloading.

    Args:
        nifti_path: Path to NIfTI file (ignored if nii_img is provided)
        slice_idx: Index of the slice to load
        axis: 'axial', 'sagittal', or 'coronal'
        nii_img: Optional pre-loaded NIfTI image object to avoid reloading

    Returns:
        numpy.ndarray: 2D slice array (copy of the slice)
    """
    if nii_img is None:
        nii_img = nib.load(nifti_path)

    data = nii_img.get_fdata()

    if axis == "axial":
        slice_data = data[:, :, slice_idx]
    elif axis == "sagittal":
        slice_data = data[slice_idx, :, :]
    elif axis == "coronal":
        slice_data = data[:, slice_idx, :]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Return a copy to avoid keeping reference to full volume
    return slice_data.copy()


def registration_qc(
    image_paths, labels, output_path, lesion_msk_path, brain_mask_path=None
):
    """
    Generate registration QC images with optimized memory usage.
    Loads only necessary slices instead of entire volumes.
    """
    # Load lesion mask header to get shape (without loading full data)
    lesion_nii = nib.load(lesion_msk_path)
    lesion_shape = lesion_nii.shape

    # Load only lesion mask for slice selection (needed for determining best slices)
    lesion_msk = lesion_nii.get_fdata()

    # Determine the slice with the largest number of positive pixels for each view
    lesion_sums_axial = np.sum(lesion_msk > 0, axis=(0, 1))
    lesion_sums_sagittal = np.sum(lesion_msk > 0, axis=(1, 2))
    lesion_sums_coronal = np.sum(lesion_msk > 0, axis=(0, 2))

    best_slice_axial = (
        np.argmax(lesion_sums_axial)
        if np.any(lesion_sums_axial > 0)
        else lesion_shape[-1] // 2
    )
    best_slice_sagittal = (
        np.argmax(lesion_sums_sagittal)
        if np.any(lesion_sums_sagittal > 0)
        else lesion_shape[0] // 2
    )
    best_slice_coronal = (
        np.argmax(lesion_sums_coronal)
        if np.any(lesion_sums_coronal > 0)
        else lesion_shape[1] // 2
    )

    # Prepare lesion mask slices (convert 0 to NaN for transparency)
    lesion_slice_axial = lesion_msk[:, :, best_slice_axial].copy()
    lesion_slice_sagittal = lesion_msk[best_slice_sagittal, :, :].copy()
    lesion_slice_coronal = lesion_msk[:, best_slice_coronal, :].copy()
    lesion_slice_axial[lesion_slice_axial == 0] = np.nan
    lesion_slice_sagittal[lesion_slice_sagittal == 0] = np.nan
    lesion_slice_coronal[lesion_slice_coronal == 0] = np.nan

    # Prepare brain mask slices (load only if needed, otherwise create from first image)
    if brain_mask_path is not None:
        # Load brain mask once and extract all slices, then free
        brain_nii = nib.load(brain_mask_path)
        brain_data = brain_nii.get_fdata()
        brain_axial = brain_data[:, :, best_slice_axial].copy()
        brain_sagittal = brain_data[best_slice_sagittal, :, :].copy()
        brain_coronal = brain_data[:, best_slice_coronal, :].copy()
        del brain_data, brain_nii  # Free full volume immediately
        brain_axial[brain_axial == 0] = np.nan
        brain_sagittal[brain_sagittal == 0] = np.nan
        brain_coronal[brain_coronal == 0] = np.nan
    else:
        # Create brain mask from first image (load once, extract slices, then free)
        first_img_nii = nib.load(image_paths[0])
        first_img_data = first_img_nii.get_fdata()
        first_img_axial = first_img_data[:, :, best_slice_axial].copy()
        first_img_sagittal = first_img_data[best_slice_sagittal, :, :].copy()
        first_img_coronal = first_img_data[:, best_slice_coronal, :].copy()
        del first_img_data, first_img_nii  # Free full volume immediately
        brain_axial = 1.0 * (first_img_axial > 0.1)
        brain_sagittal = 1.0 * (first_img_sagittal > 0.1)
        brain_coronal = 1.0 * (first_img_coronal > 0.1)
        brain_axial[brain_axial == 0] = np.nan
        brain_sagittal[brain_sagittal == 0] = np.nan
        brain_coronal[brain_coronal == 0] = np.nan
        del first_img_axial, first_img_sagittal, first_img_coronal  # Free memory

    # Free lesion mask memory (we only need the slices now)
    del lesion_msk

    # Get the number of images and set up the figure
    num_images = len(image_paths)
    num_views = 3  # Axial, sagittal, coronal
    plt.figure(figsize=(5 * num_images, 5 * num_views * 2), dpi=80, facecolor="black")
    plt.subplots_adjust(
        left=0.0001, bottom=0.001, right=0.9999, top=0.98, wspace=0.0, hspace=0.0
    )

    # Plot each view - load images slice by slice
    view_configs = [
        (best_slice_axial, "axial", "Axial", lesion_slice_axial, brain_axial),
        (
            best_slice_sagittal,
            "sagittal",
            "Sagittal",
            lesion_slice_sagittal,
            brain_sagittal,
        ),
        (best_slice_coronal, "coronal", "Coronal", lesion_slice_coronal, brain_coronal),
    ]

    # Load images once per view to avoid reloading same file multiple times
    for view_idx, (
        best_slice,
        axis,
        axis_label,
        lesion_slice,
        brain_slice,
    ) in enumerate(view_configs):
        # Pre-load all images for this view once
        view_images = {}
        for img_path in image_paths:
            nii_img = nib.load(img_path)
            data = nii_img.get_fdata()
            if axis == "axial":
                view_images[img_path] = data[:, :, best_slice].copy()
            elif axis == "sagittal":
                view_images[img_path] = data[best_slice, :, :].copy()
            elif axis == "coronal":
                view_images[img_path] = data[:, best_slice, :].copy()
            del data, nii_img  # Free full volume immediately after extracting slice

        for row in range(2):  # Two rows for each view
            for i, img_path in enumerate(image_paths):
                plt.subplot(
                    num_views * 2, num_images, (view_idx * 2 + row) * num_images + i + 1
                )

                # Use pre-loaded slice
                img_slice = view_images[img_path]

                # Apply brain mask (copy needed for in-place modification)
                img_slice_masked = img_slice.copy()
                img_slice_masked[np.isnan(brain_slice)] = np.nan

                plt.imshow(np.rot90(img_slice_masked), "gray")
                if row == 1:  # Only add lesion overlay for the second row
                    plt.imshow(
                        np.rot90(lesion_slice), "hsv", interpolation="none", alpha=0.5
                    )
                plt.axis("off")

                # Add titles only for the first row of axial images
                if view_idx == 0 and row == 0:
                    plt.title(labels[i], color="white", fontsize=14)

                # Free slice memory after use
                del img_slice_masked

        # Free all view images after processing this view
        del view_images

        # Add view label to the left side of the plot
        plt.text(
            -0.1,
            0.5 - view_idx / 2,
            axis_label,
            color="white",
            fontsize=16,
            rotation=90,
            transform=plt.gcf().transFigure,
        )

    # Show and save the figure
    plt.savefig(output_path)
    plt.close()  # Explicitly close figure to free memory
    # plt.show()


if __name__ == "__main__":
    convert_to_nii("dwi", "/home/edelarosa/Documents/datasets/dwi_dcm", "dwi")
