from PIL import Image

# List of filenames of each plot
filenames = ['cod.png', 'CTCF.png', 'DHS.png', 'EP300Conservative.png', 'h3k4me1.png','h3k4me2.png', 'h3k9me3.png', 'h3k27ac.png', 'h3k27me3.png', 'h3k36me3.png','promoter_any.png', 'promoter_forward.png']

file_path = "../../../../../scratch/ohh98/vingron/kolmogorov_smirnov/"

# Load images
images = [Image.open(file_path+filename.replace(".","_redrawn.")) for filename in filenames]

# Determine the size of each subplot (assuming they are all the same size)
width, height = images[0].size

# Create a new blank image to combine all subplots
combined_image = Image.new('RGB', (width * 4, height * 3))

# Paste each image into the combined image
for i, image in enumerate(images):
    x = (i % 4) * width
    y = (i // 4) * height
    combined_image.paste(image, (x, y))

# Save the combined image
combined_image.save('combined_plots.png')
