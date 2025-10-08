import re
import fitz
import json
import copy
import hashlib
# import tiktoken
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from src import constant
from src.fields.manual_images import ManualImages
import src.parser.image_handler as image_handler

file_path = constant.pdf_path
_min_filter_pages = 4
_max_filter_pages = 247
_page_clip = 50
_chunk_size = 256
_chunk_overlap = 50
_semantic_group_size = 10
_max_parent_size = 512


def sentence_split(text: str) -> list[str]:
    raise NotImplemented

def load_pdf() -> list[Document]:
    pdf = fitz.open(file_path)
    raw_docs = []

    for idx, page_num in enumerate(tqdm(range(len(pdf)))):
        # 过滤封面和目录
        if idx < _min_filter_pages or idx > _max_filter_pages:
            continue

        page = pdf.load_page(page_num)
        crop = fitz.Rect(0, 0, page.rect.width, page.rect.height-_page_clip)
        text = page.get_text(clip=crop)
        images = page.get_images(full=True)

        manual_images_list: List[ManualImages] = []
        for img_idx, img in enumerate(images):
            manual_image: ManualImages = image_handler.handle_images(img, img_idx, page)
            if manual_image:
                manual_images_list.append(json.loads(manual_image.model_dump_json()))

        if text.strip():
            unique_id = hashlib.md5(text.encode('utf-8')).hexdigest()
            metadata = {
                "unique_id": unique_id,
                "source": file_path,
                "page": page_num,
                "images_info": manual_images_list
            }

            raw_docs.append(metadata)

    return raw_docs