"""文本/文档内容提取：把上传文件变成归一化前的 markdown 文本。

设计原则（站在巨人肩膀上，零重依赖）：
1. 优先用 anydoc（firecrawl_anydoc，Rust 内核的 Python 绑定，cp310-abi3 通用）：
   把 doc/docx/odt/pdf/ppt/pptx/rtf/epub/xlsx/ods/odp/csv 统一转 GitHub-Flavored Markdown。
   单 .pyd 约 3.4MB，离线可用；扫描型 PDF 默认 reject（不联网，抛 NeedsOcrError）。
2. 纯文本（txt/md/markdown）不走 anydoc —— anydoc 不认这些扩展名，且纯文本
   需要的是编码自适应（UTF-8 -> GBK -> UTF-16），自写更稳。
3. 无 anydoc 时降级：docx 走标准库提取（zipfile + ElementTree），txt/md 走编码自适应，
   其余格式给出"需要 anydoc"的明确提示。
4. 输出受控子集 markdown：标题 -> #，粗体 -> **，表格 -> md 表格，
   正好被 rag/normalize.py 的归一化规则接住（粗体标题提升等）。
"""

import io
import re
import zipfile
from typing import cast
from xml.etree import ElementTree as ET

try:
    import anydoc as _anydoc
except ImportError:  # 离线包缺 anydoc 时降级，不阻断启动
    _anydoc = None

# docx 主命名空间（wordprocessingml）
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 文本解码顺序：UTF-8（标准）-> GBK（中文 Windows 常见）-> UTF-16（带 BOM）
_DECODE_ORDER = ("utf-8", "gbk", "utf-16")

# anydoc 支持的格式（Format 字面量）；纯文本不在此列，走编码自适应
_ANYDOC_FORMATS = {
    "doc", "docx", "odt", "pdf", "ppt", "pptx",
    "rtf", "epub", "xlsx", "ods", "odp", "csv",
}

# 用户友好错误映射
_ANYDOC_ERROR_HINTS = {
    "NeedsOcrError": "扫描型 PDF（纯图片页）需要 OCR，暂不支持；请先转成文字版 PDF 或 Markdown",
    "UnsupportedError": "文件格式不受 anydoc 支持",
    "MalformedError": "文件已损坏或结构异常，无法解析",
    "MissingPartError": "文件缺少关键部件，可能不是完整的该格式文件",
    "EncryptedError": "文件已加密，请先解除密码保护",
    "ResourceLimitError": "文件过大，超出解析资源限制",
    "HostedError": "托管 OCR 服务不可用（离线包不支持联网 OCR）",
}


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
    bold = run.find(f"{W}rPr/{W}b") is not None
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
    """docx 字节 -> 简易 markdown（标题/粗体/表格/段落）。无 anydoc 时的降级路径。"""
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


def _anydoc_to_markdown(data: bytes, ext: str) -> str:
    """anydoc 转换 + 错误映射为友好提示。"""
    if _anydoc is None:  # 类型收窄；调用方已保证非 None
        raise ValueError("anydoc 未加载")
    try:
        return _anydoc.to_markdown_bytes(data, format=cast("anydoc.Format", ext))
    except Exception as err:  # 统一映射为可读信息
        hint = _ANYDOC_ERROR_HINTS.get(type(err).__name__)
        if hint:
            raise ValueError(hint) from err
        # ValueError("unknown format ...") 等未映射情况
        raise ValueError(f"anydoc 转换失败（{type(err).__name__}）: {err}") from err


def extract_markdown(filename: str, data: bytes) -> str:
    """按扩展名提取 markdown 文本，供切块/向量化使用。

    路由：
      txt/md/markdown        -> 编码自适应（anydoc 不认纯文本）
      其他扩展名              -> anydoc 转 markdown（缺失时 docx 降级标准库，其余报错）
    """
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    key = "." + ext if ext else ""

    if key in {".txt", ".md", ".markdown"}:
        return decode_text(data)

    if _anydoc is not None and ext in _ANYDOC_FORMATS:
        return _anydoc_to_markdown(data, ext)

    # 降级路径：docx 标准库；其余明确提示
    if key == ".docx":
        return docx_to_markdown(data)

    if _anydoc is None:
        raise ValueError(
            "当前环境未内置文档解析组件（anydoc），仅支持 .txt / .md / .docx，"
            "请使用完整整合包"
        )
    raise ValueError(f"暂不支持 .{ext}，支持的格式：txt / md / docx / xlsx / pptx / pdf / epub 等")
