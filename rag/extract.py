"""文本/文档内容提取：把上传文件变成归一化前的 markdown 文本。

设计原则（零第三方依赖，纯标准库）：
1. docx 本质是 zip 包，正文在 word/document.xml —— zipfile + ElementTree 即可提取，
   不需要 python-docx；xlsx/pptx 同理可扩展。
2. 编码自适应：txt/md 先按 UTF-8 解，失败降级 GBK（中文 Windows 常见），再兜底
   utf-16；都不行才报错 —— 不强制用户保存成 UTF-8。
3. docx 提取输出「受控子集 markdown」：标题样式 -> #，粗体 -> **，表格 -> md 表格，
   正好被 rag/normalize.py 的归一化规则接住（粗体标题提升等）。
"""

import io
import re
import zipfile
from xml.etree import ElementTree as ET

# docx 主命名空间（wordprocessingml）
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 文本解码顺序：UTF-8（标准）-> GBK（中文 Windows 常见）-> UTF-16（带 BOM）
_DECODE_ORDER = ("utf-8", "gbk", "utf-16")


def decode_text(data: bytes) -> str:
    """按顺序尝试解码，全部失败才抛错（带清晰提示）。"""
    last_err: Exception | None = None
    for enc in _DECODE_ORDER:
        try:
            return data.decode(enc)
        except UnicodeDecodeError as err:
            last_err = err
    raise ValueError(f"无法识别文件编码（已尝试 {' / '.join(_DECODE_ORDER)}）: {last_err}")


def _run_text(run: ET.Element) -> str:
    """单个 run：文本 + 制表符，粗体包 **。"""
    if run.find(f"{W}rPr/{W}b") is not None:
        bold = True
    else:
        bold = False
    parts: list[str] = []
    for node in run:
        if node.tag == f"{W}t":
            parts.append(node.text or "")
        elif node.tag == f"{W}tab":
            parts.append("\t")
    text = "".join(parts)
    return f"**{text}**" if bold and text else text


def _para_text(p: ET.Element) -> str:
    """段落全文：遍历所有 run（含超链接内嵌），制表符转空格。"""
    runs = [_run_text(r) for r in p.iter(f"{W}r")]
    return "".join(runs).replace("\t", " ")


def _para_style(p: ET.Element) -> str:
    """标题样式 -> '#' 重复数；非标题返回空串。兼容中英文样式名。"""
    pstyle = p.find(f"{W}pPr/{W}pStyle")
    if pstyle is None:
        return ""
    val = pstyle.get(f"{W}val") or ""
    m = re.search(r"(?:Heading|标题)\s*(\d)", val)
    if not m:
        return ""
    level = int(m.group(1))
    if 1 <= level <= 6:
        return "#" * level
    return ""


def _table_markdown(tbl: ET.Element) -> str:
    """表格 -> md 表格（首行为表头 + 分隔行）。"""
    rows: list[list[str]] = []
    for tr in tbl.findall(f"{W}tr"):
        cells = [
            "".join(t.text or "" for t in tc.iter(f"{W}t")).strip() for tc in tr.findall(f"{W}tc")
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
    sep = "|" + "|".join([" --- "] * width) + "|"
    return "\n".join([lines[0], sep] + lines[1:])


def docx_to_markdown(data: bytes) -> str:
    """docx 字节 -> 简易 markdown（标题/粗体/表格/段落）。"""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as err:
        raise ValueError("不是有效的 .docx 文件（缺 word/document.xml）") from err

    body = ET.fromstring(xml).find(f"{W}body")
    if body is None:
        raise ValueError("docx 正文为空")

    out: list[str] = []
    for child in body:
        tag = child.tag
        if tag == f"{W}p":
            style = _para_style(child)
            text = _para_text(child)
            if not text:
                continue
            out.append(f"{style} {text}" if style else text)
        elif tag == f"{W}tbl":
            table = _table_markdown(child)
            if table:
                out.append(table)
        # sectPr（节属性）等其他节点忽略
    return "\n\n".join(out)


# 上传文件扩展名 -> 提取函数；不在列表里的一律报"暂不支持"
EXTRACTORS = {
    ".docx": docx_to_markdown,
    # 纯文本走编码自适应
    ".txt": lambda data: decode_text(data),
    ".md": lambda data: decode_text(data),
    ".markdown": lambda data: decode_text(data),
}


def extract_markdown(filename: str, data: bytes) -> str:
    """按扩展名提取 markdown 文本，供切块/向量化使用。"""
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    key = "." + ext if ext else ""
    if key not in EXTRACTORS:
        raise ValueError(f"暂支持 {' / '.join(sorted(EXTRACTORS))}，其他格式请先转换")
    return EXTRACTORS[key](data)
