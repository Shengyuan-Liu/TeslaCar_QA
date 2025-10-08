from typing import Optional
from pydantic import BaseModel, Field

class ManualImages(BaseModel):
    """
    用户手册图片元数据模型

    用于存储Tesla用户手册中提取的图片相关信息，支持MongoDB持久化存储。
    该模型记录图片所在页码、文件系统存储路径以及图片相关的标题或说明文字。

    Attributes:
        page (Optional[int]): 图片所在页码，从1开始计数，必须≥1
        image_path (Optional[str]): 图片在文件系统中的存储路径，非空字符串
        title (Optional[str]): 图片相关的标题或说明内容，多个文本区块用换行符连接
    """
    page: Optional[int] = Field(ge=1, description='页码从1开始')
    image_path: Optional[str] = Field(min_length=1, description='图片储存路径')
    title: Optional[str] = Field(description='标题内容, 多个区块用换行符连接')