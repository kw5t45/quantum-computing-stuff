from pathlib import Path
from PIL import Image

# Make images greyscale and convert to png, if not it might cause errors later in nd library. also assert square images
def convert_folder_to_grayscale_png(folder_path):
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    for file in folder.iterdir():
        if not file.is_file():
            continue

        try:
            with Image.open(file) as img:
                width, height = img.size

                assert width == height, (
                    f"Image is not square: {file.name} "
                    f"({width}x{height})"
                )

                gray = img.convert("L")

                output_path = file.with_suffix(".png")
                gray.save(output_path)

                print(f"Converted: {file.name} -> {output_path.name}")

        except Exception as e:
            print(f"Skipping {file.name}: {e}")


convert_folder_to_grayscale_png(fr"resources/test_images")