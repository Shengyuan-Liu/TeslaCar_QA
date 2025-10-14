from src.parser.image_handler import handle_images
from src.parser.pdf_parse import load_pdf
import fitz


pdf = fitz.open("data/Tesla_Manual.pdf")


from src.fields.manual_images import ManualImages

a = ManualImages(
    page=1,
    image_path="111.111",
    title="11111"
)

print(load_pdf())