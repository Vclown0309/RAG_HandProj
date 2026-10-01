/* 轻量 Markdown 渲染器（演示页专用子集）
 *
 * 覆盖：标题、粗体/斜体、行内代码、代码块、无序/有序列表、表格、
 *       引用、段落。零依赖、离线可用（本地文件，非 CDN）。
 *
 * 安全：所有文本先 HTML 转义再渲染标记；链接仅放行 http/https，
 *       javascript: 等协议一律降级为纯文本。
 */
(function () {
  'use strict';

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // 行内渲染：escape 之后处理标记
  function inline(html) {
    // 链接：仅 http/https 协议（转义后 href 无注入风险）
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // 代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 粗体
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // 斜体
    html = html.replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>');
    return html;
  }

  function isTableRow(s) { return s.charAt(0) === '|'; }
  function isTableSep(s) { return /^\s*\|?[\s:|-]+\|?\s*$/.test(s) && s.indexOf('-') !== -1 && s.indexOf('|') !== -1; }

  function renderMarkdown(text) {
    var lines = String(text).split('\n');
    var out = [];
    var p = [];            // 当前段落缓冲
    var listType = null;   // 'ul' | 'ol'
    var inCode = false;
    var codeBuf = [];
    var tbl = [];

    function flushPara() {
      if (p.length) {
        out.push('<p>' + inline(p.join(' ')) + '</p>');
        p = [];
      }
    }
    function closeList() {
      if (listType) { out.push('</' + listType + '>'); listType = null; }
    }
    function flushTable() {
      if (!tbl.length) return;
      // 孤立表格行（无分隔行/无后续行）不是表格：降级为普通段落
      if (tbl.length === 1) {
        out.push('<p>' + inline(escapeHtml(tbl[0])) + '</p>');
        tbl = [];
        return;
      }
      var isHeader = tbl.length >= 2 && isTableSep(tbl[1]);
      var start = isHeader ? 2 : 1;
      var html = '<table>';
      if (isHeader) {
        html += '<thead><tr>' + tbl[0].split('|').slice(1, -1).map(function (c) {
          return '<th>' + inline(c.trim()) + '</th>';
        }).join('') + '</tr></thead>';
      }
      html += '<tbody>';
      var i;
      for (i = start; i < tbl.length; i++) {
        html += '<tr>' + tbl[i].split('|').slice(1, -1).map(function (c) {
          return '<td>' + inline(c.trim()) + '</td>';
        }).join('') + '</tr>';
      }
      html += '</tbody></table>';
      out.push(html);
      tbl = [];
    }

    var i;
    for (i = 0; i < lines.length; i++) {
      var raw = lines[i];
      var s = raw.trim();

      // 代码块
      if (/^```/.test(s)) {
        flushPara(); closeList(); flushTable();
        if (!inCode) { inCode = true; codeBuf = []; }
        else { out.push('<pre><code>' + escapeHtml(codeBuf.join('\n')) + '</code></pre>'); inCode = false; }
        continue;
      }
      if (inCode) { codeBuf.push(raw); continue; }

      // 空行：段落/列表/表格边界
      if (!s) {
        flushPara(); closeList(); flushTable();
        continue;
      }

      // 表格行
      if (isTableRow(s)) {
        flushPara(); closeList();
        tbl.push(s);
        continue;
      }
      flushTable();

      // 标题
      var hm = /^(#{1,6})\s+(.*)$/.exec(s);
      if (hm) {
        flushPara(); closeList();
        var lvl = hm[1].length;
        out.push('<h' + lvl + '>' + inline(escapeHtml(hm[2])) + '</h' + lvl + '>');
        continue;
      }

      // 引用
      if (/^>\s?/.test(s)) {
        flushPara(); closeList();
        out.push('<blockquote>' + inline(escapeHtml(s.replace(/^>\s?/, ''))) + '</blockquote>');
        continue;
      }

      // 无序列表
      var um = /^[-*+]\s+(.*)$/.exec(s);
      if (um) {
        flushPara();
        if (listType !== 'ul') { closeList(); out.push('<ul>'); listType = 'ul'; }
        out.push('<li>' + inline(escapeHtml(um[1])) + '</li>');
        continue;
      }
      // 有序列表
      var om = /^\d+\.\s+(.*)$/.exec(s);
      if (om) {
        flushPara();
        if (listType !== 'ol') { closeList(); out.push('<ol>'); listType = 'ol'; }
        out.push('<li>' + inline(escapeHtml(om[1])) + '</li>');
        continue;
      }
      closeList();

      // 普通段落行
      p.push(escapeHtml(raw));
    }

    flushPara(); closeList(); flushTable();
    if (inCode) out.push('<pre><code>' + escapeHtml(codeBuf.join('\n')) + '</code></pre>');
    return out.join('');
  }

  window.renderMarkdown = renderMarkdown;
})();
