from typing import Optional
from pydantic import BaseModel, Field

class ManualInfo(BaseModel):
    unique_id: str = Field(description="唯一标识符")
    metadata: dict = Field(description="储存文档的meta信息")
    page_content: Optional[str] = Field(description="文档分片的内容")