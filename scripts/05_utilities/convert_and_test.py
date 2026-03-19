import SimpleITK as sitk
import os

input_folder = "/Users/acchuaccount/CSS483/CT-Scan/HN-CHUM-030"
output_file = "/Users/acchuaccount/CSS483/CT-Scan/HN-CHUM-030.nrrd"

# get sorted list of PNGs
png_files = sorted([os.path.join(input_folder, f) 
                    for f in os.listdir(input_folder) if f.endswith(".png")])

# read slices and stack into 3D volume
slices = [sitk.ReadImage(f) for f in png_files]
volume = sitk.JoinSeries(slices)

# write to NRRD
sitk.WriteImage(volume, output_file)
print("Saved:", output_file)