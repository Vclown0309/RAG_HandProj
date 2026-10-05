"""演示页静态自测：代码块样式、离线依赖、页面可达性。

M5 回归用的 UI 检查：
    1. 代码块纯色高对比（无 hljs 残留、无深色底上深色字的覆盖规则）
    2. 代码块不折行（white-space: pre，无 pre-wrap 覆盖）
    3. 页面零 CDN 依赖（vendor 全本地）
    4. 页面可访问 + 关键结构锚点存在
"""

import re
from pathlib import Path

from fastapi.testclient import TestClient

import main as main_module

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / 'static' / 'index.html'
VENDOR = ROOT / 'static' / 'vendor'


def _contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 对比度：值 >= 4.5 为 AA（正常文本）。"""

    def lum(hex_color: str) -> float:
        rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            for c in rgb
        ]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    l1, l2 = lum(fg), lum(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def test_index_has_no_hljs_residue() -> None:
    """已撤掉语法高亮：页面与 vendor 均无 hljs 残留。"""
    html = INDEX.read_text(encoding='utf-8')
    assert 'highlight' not in html, 'index.html 仍有 hljs 引用'
    assert 'github-dark' not in html, 'index.html 仍有 hljs 主题引用'
    assert not any(p.name.startswith('highlight') for p in VENDOR.iterdir()), 'vendor 仍有 hljs 文件'
    assert not any(p.name.startswith('powershell') for p in VENDOR.iterdir()), 'vendor 仍有 powershell 语言包'
    assert not any(p.name.startswith('github-dark') for p in VENDOR.iterdir()), 'vendor 仍有 hljs 主题'


def test_code_block_style_high_contrast() -> None:
    """代码块：深底 + 纯白字，无深色字覆盖规则（此前 color:#333 残留翻车）。"""
    css = INDEX.read_text(encoding='utf-8')
    pre_block = re.search(
        r'#answer-area pre, \.source-body pre \{[^}]*\}', css, re.DOTALL
    )
    assert pre_block, '统一 pre 样式规则缺失'
    rule = pre_block.group(0)
    assert 'white-space: pre' in rule, '必须不折行（white-space: pre）'
    assert 'pre-wrap' not in rule, '统一规则里出现折行'
    assert 'color: #ffffff' in rule, '代码文字必须是高对比纯白'
    bg = re.search(r'background: (#[0-9a-fA-F]{6})', rule).group(1)  # type: ignore[union-attr]
    ratio = _contrast_ratio('#ffffff', bg)
    assert ratio >= 7, f'对比度不足：{ratio:.1f}:1（应 >= 7:1 高对比）'


def test_no_legacy_pre_wrap_override() -> None:
    """所有 pre 规则统一：不折行 + 无深灰字残留，无旧规则覆盖。"""
    css = INDEX.read_text(encoding='utf-8')
    # 抽出全部 {rule} 块，筛选择器含 pre 的规则
    rules = re.findall(r'([^{}]+)\{([^{}]*)\}', css)
    pre_rules = [(sel.strip(), body.strip()) for sel, body in rules if 'pre' in sel]
    assert pre_rules, '未找到任何 pre 规则'
    # 主容器规则（选择器含 pre 且不是 code 子元素）必须显式不折行
    container_rules = [b for s, b in pre_rules if 'pre code' not in s]
    assert any('white-space: pre' in b for b in container_rules), 'pre 容器规则缺少不折行'
    for sel, body in pre_rules:
        assert 'pre-wrap' not in body, f'{sel} 出现折行（pre-wrap）'
        assert '#333' not in body, f'{sel} 出现深灰代码字'


def test_vendor_local_only() -> None:
    """零 CDN：页面脚本/样式全部指向本地 /static/vendor。"""
    html = INDEX.read_text(encoding='utf-8')
    refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert refs, '未找到本地资源引用'
    for ref in refs:
        path = ROOT / ref.lstrip('/')
        assert path.exists(), f'本地资源缺失：{ref}'


def test_page_reachable() -> None:
    """页面可达 + 关键结构锚点存在。"""
    client = TestClient(main_module.app)
    resp = client.get('/')
    assert resp.status_code == 200
    assert 'RAG 演示台' in resp.text
    for anchor in ('ask-form', 'sources-section', 'library-section'):
        assert anchor in resp.text, f'缺少结构锚点：{anchor}'
