# mypy: ignore-errors
# 工具脚本：运行期依赖 PIL/pptx/fpdf/pymupdf 等无类型 stub 的库（uv --with 按需注入），
# 全量类型检查无价值；已按 ruff 规范保持代码质量。

"""生成 RAG 工程横向对比用典例文档集（bench_corpus/）。

设计原则：六类格式内容同源（同一套 RAG 知识点），仅按格式特性变排版——
横向对比时内容变量被控住，测出的差异就是「格式」本身对
anydoc → Markdown → 切块 → 向量化检索 的影响。

知识点与判定锚句（6 个，覆盖六节内容，测试集据此判定 top-3 命中）：
    开卷考试 / 一本正经地胡说 / 标题结构切 / 方向才表达语义 / 余弦相似度 / Markdown

用法（依赖按需注入，不写进 pyproject）：
    uv run --with python-docx --with openpyxl --with python-pptx \
        --with fpdf2 --with pillow python -m scripts.make_bench_corpus
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'bench_corpus'
FONT_HEI = r'C:\Windows\Fonts\simhei.ttf'
FONT_FANG = r'C:\Windows\Fonts\simfang.ttf'

# ---------------------------------------------------------------- 内容同源定义

SECTIONS: list[dict[str, str]] = [
    {
        'title': 'RAG 是什么',
        'body': '检索增强生成（Retrieval-Augmented Generation）：回答前先从你的私有资料里查资料，把查到的原文塞进提问，再让模型基于资料回答。类比：开卷考试——裸问答是闭卷，模型只会背书上的还可能记错；RAG 是开卷，带着笔记进场翻到相关段落再答题。',
        'anchor': '开卷考试',
    },
    {
        'title': '为什么要先查资料',
        'body': '模型的记忆是训练时写死的，有两个硬伤：不知道你的事（你的笔记、公司文档模型从没见过）；会一本正经地胡说（记忆模糊时倾向编一个听起来合理的答案，这就是幻觉）。RAG 的思路是不改造模型，而是在提问时喂它需要的那几页资料——模型不擅长记忆，但擅长照着资料说话。',
        'anchor': '一本正经地胡说',
    },
    {
        'title': '切块：两种切法',
        'body': '模型每次能看的文字有限，先把文档切成小块，提问时只挑相关的几块。窗口切：固定 500 字一块、步长 450，重叠 50 字防止一句话被从中间切断。标题结构切：按文档标题分节，一个标题一块，标题是天然的语义边界。项目用两级策略：标题分节为主，某节超过 500 字在节内再窗口切；无标题文档退化为纯窗口切。',
        'anchor': '标题结构切',
    },
    {
        'title': '向量化：文字变数字',
        'body': '电脑不懂意思但懂数字。嵌入模型把一段文字映射成一串数字（向量），规则是意思相近的文字向量方向也相近：猫和猫咪的向量挨得很近，猫和汽车的向量离得很远。向量是一串数字，数字本身没意义，方向才表达语义。',
        'anchor': '方向才表达语义',
    },
    {
        'title': '检索：余弦相似度',
        'body': '提问来了：先把问题也变成向量，然后和库里每一块的向量比方向像不像，取最像的 top-k 块。数学上就是余弦相似度：点积衡量方向一致性，除以模长抵消长度影响，范围负一到一，越大越像。为什么用余弦不用距离？向量长度受文本长度影响，余弦只看方向更公平。',
        'anchor': '余弦相似度',
    },
    {
        'title': '最佳输入格式',
        'body': '向量化用什么格式输入最佳？Markdown。理由：结构完整，标题、表格、代码块都是切块与检索的强信号；纯文本会把这些结构抹平，PDF 表格转纯文本后语义粘连。让资料保持 Markdown 进管道，检索命中率和答案可追溯性都更好。',
        'anchor': 'Markdown',
    },
]

ROUTES: list[tuple[str, str, str]] = [
    ('微调', '高：要数据、要算力、要调参', '慢，像重新训练一次'),
    ('蒸馏', '高：需要大模型老师 + 大量标注', '慢，接近研发投入'),
    ('RAG', '低：代码 + 本地模型即可', '快：上传文档就能用，效果立竿见影'),
]

PROMPT_TEMPLATE = """资料1：...
资料2：...
问题：...
请仅依据以上资料回答，不要编造资料外的内容。"""

# ---------------------------------------------------------------- 图片

def make_images() -> dict[str, Path]:
    """生成两张示意小图：RAG 流程 + 余弦示意。返回 {key: path}。"""
    img_dir = OUT / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_HEI, 22)

    # RAG 流程：文档 → 切块 → 向量 → 检索 → 拼Prompt → 回答
    steps = ['文档', '切块', '向量', '检索', '拼Prompt', '回答']
    w, h = 780, 190
    img = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(img)
    bw, bh, gap = 100, 60, 40
    y = (h - bh) // 2
    x = 40
    for i, s in enumerate(steps):
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=10, outline='#2563EB', width=2)
        d.text((x + bw / 2, y + bh / 2), s, font=font, fill='#111827', anchor='mm')
        if i < len(steps) - 1:
            ax, ay = x + bw + gap // 2, y + bh // 2
            d.line([(ax - 14, ay), (ax + 14, ay)], fill='#2563EB', width=3)
            d.polygon([(ax + 14, ay), (ax + 6, ay - 8), (ax + 6, ay + 8)], fill='#2563EB')
        x += bw + gap
    flow = img_dir / 'rag_flow.png'
    img.save(flow)
    return {'flow': flow}


# ---------------------------------------------------------------- Word

def make_word() -> Path:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    # 中文字体
    style = doc.styles['Normal']
    style.font.name = '仿宋'
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '仿宋')

    doc.add_heading('RAG 应用开发入门（典例版）', level=0)
    doc.add_paragraph('—— Word 格式典例：正式文档排版，用于 anydoc 转换对比。')

    for sec in SECTIONS:
        doc.add_heading(sec['title'], level=1)
        p = doc.add_paragraph(sec['body'])

    doc.add_heading('路线对比', level=1)
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    for i, head in enumerate(['路线', '成本', '见效']):
        table.rows[0].cells[i].text = head
    for route in ROUTES:
        cells = table.add_row().cells
        for i, v in enumerate(route):
            cells[i].text = v

    doc.add_heading('Prompt 模板', level=1)
    for line in PROMPT_TEMPLATE.splitlines():
        p = doc.add_paragraph(line)
        p.paragraph_format.left_indent = Pt(10)

    p = doc.add_heading('', level=1)
    doc.add_paragraph('（本段为补充段落：RAG 全链路源码位于 rag 包，演示页可点开参考源核对答案依据。）')

    path = OUT / '01_word_rag入门.docx'
    doc.save(path)
    return path


# ---------------------------------------------------------------- Excel

def make_excel() -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()

    ws = wb.active
    ws.title = '知识点速查'
    ws.append(['主题', '一句话', '类比', '判定锚句'])
    for sec in SECTIONS:
        ws.append([sec['title'], sec['body'][:40], sec['body'][:20], sec['anchor']])
    for col, width in zip('ABCD', [16, 60, 30, 14]):
        ws.column_dimensions[col].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='D9E2F3')

    ws2 = wb.create_sheet('路线对比')
    ws2.append(['路线', '成本', '见效'])
    for row in ROUTES:
        ws2.append(list(row))
    for col, width in zip('ABC', [10, 40, 40]):
        ws2.column_dimensions[col].width = width
    for cell in ws2[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill('solid', fgColor='D9E2F3')
    for row in ws2.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical='center')

    path = OUT / '02_excel_rag知识点.xlsx'
    wb.save(path)
    return path


# ---------------------------------------------------------------- PPT

def make_ppt(flow: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    blank = prs.slide_layouts[6]

    # 封面
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(2))
    tf = tb.text_frame
    tf.text = 'RAG 应用开发入门（典例版）'
    tf.paragraphs[0].font.size = Pt(44)
    tf.paragraphs[0].font.bold = True
    sub = s.shapes.add_textbox(Inches(1), Inches(4.2), Inches(11), Inches(1))
    sub.text_frame.text = '—— PPT 格式典例：标题 + 要点 + 图示，用于 anydoc 转换对比'
    sub.text_frame.paragraphs[0].font.size = Pt(20)

    # 内容页
    for sec in SECTIONS:
        s = prs.slides.add_slide(blank)
        title = s.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(1))
        title.text_frame.text = sec['title']
        title.text_frame.paragraphs[0].font.size = Pt(32)
        title.text_frame.paragraphs[0].font.bold = True
        body = s.shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(12), Inches(5))
        body.text_frame.word_wrap = True
        # 要点化：按句号拆 3-4 条
        points = [p.strip() for p in sec['body'].replace('；', '。').split('。') if p.strip()][:4]
        tf = body.text_frame
        tf.text = points[0]
        for p in points[1:]:
            tf.add_paragraph().text = p
        for p in tf.paragraphs:
            p.font.size = Pt(18)

    # 流程示意页
    s = prs.slides.add_slide(blank)
    title = s.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12), Inches(1))
    title.text_frame.text = 'RAG 全链路示意'
    title.text_frame.paragraphs[0].font.size = Pt(32)
    title.text_frame.paragraphs[0].font.bold = True
    s.shapes.add_picture(str(flow), Inches(0.8), Inches(2.0), width=Inches(11.5))

    path = OUT / '03_ppt_rag入门.pptx'
    prs.save(path)
    return path


# ---------------------------------------------------------------- EPUB

def make_epub(flow: Path) -> Path:
    """手写 EPUB3 结构（zipfile）：封面图 + 章节 + 内嵌流程示意图。"""
    epub_dir = OUT / '_epub_build'
    oebps = epub_dir / 'OEBPS'
    (oebps / 'images').mkdir(parents=True, exist_ok=True)

    # 内嵌图片（拷贝流程图为内嵌资源）
    from shutil import copyfile
    copyfile(flow, oebps / 'images' / 'rag_flow.png')

    (oebps / 'ch1.xhtml').write_text(_xhtml('第一章：RAG 与开卷考试', SECTIONS[0]['body'] + SECTIONS[1]['body']), encoding='utf-8')
    (oebps / 'ch2.xhtml').write_text(_xhtml('第二章：切块与向量化', SECTIONS[2]['body'] + SECTIONS[3]['body']), encoding='utf-8')
    (oebps / 'ch3.xhtml').write_text(_xhtml('第三章：检索与最佳格式', SECTIONS[4]['body'] + SECTIONS[5]['body'], with_image=True), encoding='utf-8')
    (oebps / 'content.opf').write_text(_opf(), encoding='utf-8')
    (oebps / 'toc.ncx').write_text(_ncx(), encoding='utf-8')
    (epub_dir / 'META-INF').mkdir(exist_ok=True)
    (epub_dir / 'META-INF' / 'container.xml').write_text(
        '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
        '</rootfiles></container>', encoding='utf-8')

    path = OUT / '04_epub_rag图文.epub'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
        for f in sorted(epub_dir.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(epub_dir).as_posix())
    return path


def _xhtml(title: str, body: str, with_image: bool = False) -> str:
    img = '<p><img src="images/rag_flow.png" alt="RAG 流程示意"/></p>' if with_image else ''
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head>'
        f'<title>{title}</title></head><body>'
        f'<h1>{title}</h1><p>{body}</p>{img}'
        '</body></html>'
    )


def _opf() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<dc:identifier id="uid">bench-rag-001</dc:identifier>'
        '<dc:title>RAG 应用开发入门（典例版）</dc:title>'
        '<dc:creator>RAG_HandProj</dc:creator>'
        '<dc:language>zh-CN</dc:language>'
        '</metadata>'
        '<manifest>'
        '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="ch3" href="ch3.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
        '<item id="img" href="images/rag_flow.png" media-type="image/png"/>'
        '</manifest>'
        '<spine toc="ncx"><itemref idref="ch1"/><itemref idref="ch2"/><itemref idref="ch3"/></spine>'
        '</package>'
    )


def _ncx() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        '<head><meta name="dtb:uid" content="bench-rag-001"/></head>'
        '<docTitle><text>RAG 应用开发入门（典例版）</text></docTitle>'
        '<navMap>'
        '<navPoint id="n1" playOrder="1"><navLabel><text>第一章</text></navLabel><content src="ch1.xhtml"/></navPoint>'
        '<navPoint id="n2" playOrder="2"><navLabel><text>第二章</text></navLabel><content src="ch2.xhtml"/></navPoint>'
        '<navPoint id="n3" playOrder="3"><navLabel><text>第三章</text></navLabel><content src="ch3.xhtml"/></navPoint>'
        '</navMap></ncx>'
    )


# ---------------------------------------------------------------- Markdown

def make_md() -> Path:
    lines = ['# RAG 应用开发入门（典例版）', '', '—— Markdown 格式典例：原生结构，作为对照基准。', '']
    for sec in SECTIONS:
        lines += [f'## {sec["title"]}', '', sec['body'], '']
    lines += ['## 路线对比', '', '| 路线 | 成本 | 见效 |', '|---|---|---|']
    for r in ROUTES:
        lines += [f'| {r[0]} | {r[1]} | {r[2]} |']
    lines += ['', '## Prompt 模板', '', '```', PROMPT_TEMPLATE, '```', '']
    path = OUT / '05_md_rag入门.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


# ---------------------------------------------------------------- PDF

def make_pdf() -> Path:
    from fpdf import FPDF

    pdf = FPDF(format='A4')
    pdf.add_font('hei', '', FONT_HEI)
    pdf.add_font('fang', '', FONT_FANG)
    pdf.set_auto_page_break(True, margin=20)

    pdf.add_page()
    pdf.set_font('hei', '', 22)
    pdf.cell(0, 14, 'RAG 应用开发入门（典例版）', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('fang', '', 11)
    pdf.cell(0, 8, '—— PDF 格式典例：正式文档排版，用于 anydoc 转换对比', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(6)

    for sec in SECTIONS:
        pdf.set_font('hei', '', 15)
        pdf.cell(0, 10, sec['title'], new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('fang', '', 11)
        pdf.multi_cell(0, 7, sec['body'])
        pdf.ln(3)

    pdf.set_font('hei', '', 15)
    pdf.cell(0, 10, '路线对比', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('fang', '', 11)
    for route in ROUTES:
        pdf.cell(0, 7, f'【{route[0]}】成本：{route[1]}；见效：{route[2]}', new_x='LMARGIN', new_y='NEXT')

    pdf.set_font('hei', '', 15)
    pdf.cell(0, 10, 'Prompt 模板', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('hei', '', 10)
    for line in PROMPT_TEMPLATE.splitlines():
        pdf.cell(0, 7, line, new_x='LMARGIN', new_y='NEXT')

    path = OUT / '06_pdf_rag入门.pdf'
    pdf.output(str(path))
    return path


# ---------------------------------------------------------------- 扫描型 PDF（图片型）

def make_scan_pdf() -> Path:
    """文本 PDF → 逐页光栅化 → 图片型 PDF（模拟扫描件）。

    内容与 06 文本版完全一致，仅载体从文字变图像，用于测 anydoc 的 OCR 链路损耗。
    依赖 pymupdf：uv run --with pymupdf
    """
    import pymupdf

    src = OUT / '06_pdf_rag入门.pdf'
    tmp_pngs: list[Path] = []
    with pymupdf.open(src) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            png = OUT / 'images' / f'_scan_p{i}.png'
            pix.save(str(png))
            tmp_pngs.append(png)

    from PIL import Image, ImageOps

    imgs = [ImageOps.grayscale(Image.open(p).convert('RGB')) for p in tmp_pngs]
    path = OUT / '07_pdf扫描版_rag入门.pdf'
    imgs[0].save(path, save_all=True, append_images=imgs[1:])
    for p in tmp_pngs:
        p.unlink()
    return path


# ---------------------------------------------------------------- 设计说明

def make_readme() -> Path:
    lines = [
        '# 典例文档集 · 设计说明（bench_corpus）',
        '',
        '用途：横向对比「文档格式 → anydoc → Markdown → 切块 → 向量化检索」的质量差异。',
        '',
        '## 设计原则：内容同源，仅格式变排版',
        '',
        '六类文档都围绕同一套 RAG 知识点（六节），只按格式特性排版——',
        'word 正式排版、excel 表格化、ppt 标题要点页、epub 章节图文、md 原生、pdf 正式文档、',
        'pdf 扫描版（文本版的光栅化图片件）。',
        '这样横向对比时「内容变量」被控住，测出的差异就是格式本身的影响。',
        '',
        '## 典例清单',
        '',
        '| 文件 | 类型 | 说明 |',
        '|---|---|---|',
        '| 01_word_rag入门.docx | Word | 正式排版：Heading + 表格 + 代码块 |',
        '| 02_excel_rag知识点.xlsx | Excel | 表格化：知识点速查 + 路线对比两张表 |',
        '| 03_ppt_rag入门.pptx | PPT | 封面 + 6 内容页 + 1 流程图页 |',
        '| 04_epub_rag图文.epub | EPUB | 3 章节 + 内嵌流程示意图 |',
        '| 05_md_rag入门.md | Markdown | 原生基准（对照） |',
        '| 06_pdf_rag入门.pdf | PDF | 文本型正式文档 |',
        '| 07_pdf扫描版_rag入门.pdf | PDF | 扫描型（图片型，06 的光栅化） |',
        '',
        '## 六节知识点与判定锚句',
        '',
        '| 节 | 锚句 |',
        '|---|---|',
    ]
    for sec in SECTIONS:
        lines.append(f'| {sec["title"]} | {sec["anchor"]} |')
    lines += [
        '',
        '## 测试流程（建议）',
        '',
        '1. anydoc 把每份典例转成 Markdown（记录转换告警/丢失项）',
        '2. 转换结果用 chunk_markdown 切块，每类文档得到若干块',
        '3. 六题提问（锚句即答案所在块），逐类文档测 top-3 命中率',
        '4. 记录：块数、总字数、命中率、转换耗时 → 产出对比表',
        '',
        '## 预期观察点',
        '',
        '- excel 表格转 md 后是否保留表头/行结构，还是粘连成一坨文本',
        '- ppt 要点页转 md 后标题层级是否保留',
        '- epub 图文混排：图片是否丢失、章节结构是否保留',
        '- pdf 表格/代码块是否被截断或粘连',
        '- md 作为基准：原生结构应保持最高命中率（若否，说明 anydoc 或切块有问题）',
        '',
        '## 生成方式（可复现）',
        '',
        '```',
        'uv run --with python-docx --with openpyxl --with python-pptx --with fpdf2 --with pillow python -m scripts.make_bench_corpus',
        '```',
        '',
        '依赖全部按需注入，不写进 pyproject（典例生成是实验工具，不是项目运行时依赖）。',
        '',
        '## 测试边界说明',
        '',
        '横向对比与挡位推荐基于 x64 Windows 实测（CPU 版 / Vulkan 核显版 / CUDA 独显版），',
        '未测试 ROCm（AMD 独显 Linux）与 arm 架构 CPU。',
        '',
    ]
    path = OUT / 'README.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def main() -> None:
    try:
        OUT.mkdir(exist_ok=True)
        images = make_images()
        files = [
            make_word(),
            make_excel(),
            make_ppt(images['flow']),
            make_epub(images['flow']),
            make_md(),
            make_pdf(),
            make_scan_pdf(),
            make_readme(),
        ]
        for f in files:
            print(f'{f.name:28s} {f.stat().st_size / 1024:.1f} KB')
    except Exception:
        import traceback

        (OUT / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        print('ERROR logged to bench_corpus/error.log')
        raise


if __name__ == '__main__':
    main()
