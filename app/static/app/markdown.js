// A small markdown renderer that builds DOM nodes, never HTML strings.
//
// Run READMEs are untrusted text as far as the browser is concerned: every
// piece of content goes through createTextNode/textContent, so a README that
// contains "<script>" shows those characters and nothing more. Supported:
// ATX headings, fenced code, tables, ordered/unordered lists (one level of
// nesting by indent), blockquotes, rules, paragraphs; inline code, bold,
// italic and links (http(s) only, or resolved by the caller).
"use strict";

(function () {
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  // -- inline ---------------------------------------------------------------
  // Order matters: code spans first, so nothing inside them is interpreted.
  const INLINE_SOURCE = /(`+)([^`]|[^`][\s\S]*?[^`])\1|\*\*([^*]+?)\*\*|__([^_]+?)__|\[([^\]]+)\]\(([^)\s]+)\)|(^|[^\w*])\*([^*\s][^*]*?)\*(?!\w)|(^|[^\w])_([^_\s][^_]*?)_(?!\w)/g;

  function inline(parent, text, opts) {
    // A fresh regex per call: inline() recurses for bold, italic and link
    // labels, and a shared /g regex would have its lastIndex reset under it.
    const INLINE = new RegExp(INLINE_SOURCE.source, "g");
    let last = 0;
    let m;
    while ((m = INLINE.exec(text)) !== null) {
      let start = m.index;
      // The italic alternatives capture one leading boundary character.
      const lead = m[7] !== undefined ? m[7] : m[9] !== undefined ? m[9] : "";
      start += lead.length;
      if (start > last) parent.appendChild(document.createTextNode(text.slice(last, start)));
      if (m[2] !== undefined) {
        parent.appendChild(el("code", null, m[2]));
      } else if (m[3] !== undefined || m[4] !== undefined) {
        const strong = el("strong");
        inline(strong, m[3] !== undefined ? m[3] : m[4], opts);
        parent.appendChild(strong);
      } else if (m[5] !== undefined) {
        parent.appendChild(link(m[5], m[6], opts));
      } else if (m[8] !== undefined || m[10] !== undefined) {
        const em = el("em");
        inline(em, m[8] !== undefined ? m[8] : m[10], opts);
        parent.appendChild(em);
      }
      last = INLINE.lastIndex;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  function link(label, href, opts) {
    let url = null;
    if (/^https?:\/\//i.test(href)) url = href;
    else if (opts && opts.resolveLink) url = opts.resolveLink(href);
    if (!url) {
      // A link we cannot follow safely (a repo-relative path, a javascript:
      // URL): keep the words, drop the link.
      const span = el("span", "md-ref");
      inline(span, label, opts);
      span.title = href;
      return span;
    }
    const a = el("a");
    a.href = url;
    if (/^https?:\/\//i.test(url)) { a.target = "_blank"; a.rel = "noopener noreferrer"; }
    inline(a, label, opts);
    return a;
  }

  // -- blocks ---------------------------------------------------------------
  const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/;
  const FENCE = /^\s*(```|~~~)/;
  const LIST = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
  const RULE = /^\s*([-*_])(\s*\1){2,}\s*$/;
  const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function cells(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
  }

  function render(source, opts) {
    const root = document.createDocumentFragment();
    const lines = String(source || "").replace(/\r\n?/g, "\n").split("\n");
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      if (!line.trim()) { i++; continue; }

      if (FENCE.test(line)) {
        const fence = line.trim().slice(0, 3);
        const body = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(fence)) body.push(lines[i++]);
        i++;
        const pre = el("pre");
        pre.appendChild(el("code", null, body.join("\n")));
        root.appendChild(pre);
        continue;
      }

      const h = HEADING.exec(line);
      if (h) {
        const node = el("h" + h[1].length);
        inline(node, h[2], opts);
        root.appendChild(node);
        i++;
        continue;
      }

      if (RULE.test(line)) { root.appendChild(el("hr")); i++; continue; }

      if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
        const table = el("table");
        const head = el("thead");
        const tr = el("tr");
        for (const c of cells(line)) { const th = el("th"); inline(th, c, opts); tr.appendChild(th); }
        head.appendChild(tr);
        table.appendChild(head);
        const body = el("tbody");
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          const row = el("tr");
          for (const c of cells(lines[i])) { const td = el("td"); inline(td, c, opts); row.appendChild(td); }
          body.appendChild(row);
          i++;
        }
        table.appendChild(body);
        const wrap = el("div", "scroll-x");
        wrap.appendChild(table);
        root.appendChild(wrap);
        continue;
      }

      if (/^\s*>/.test(line)) {
        const body = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ""));
        const quote = el("blockquote");
        quote.appendChild(render(body.join("\n"), opts));
        root.appendChild(quote);
        continue;
      }

      if (LIST.test(line)) {
        i = renderList(root, lines, i, opts);
        continue;
      }

      // Paragraph: until a blank line or the start of another block.
      const para = [];
      while (i < lines.length && lines[i].trim() && !FENCE.test(lines[i]) && !HEADING.test(lines[i])
             && !LIST.test(lines[i]) && !/^\s*>/.test(lines[i]) && !RULE.test(lines[i])
             && !(lines[i].includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]))) {
        para.push(lines[i].trim());
        i++;
      }
      const p = el("p");
      // A trailing double space or backslash is a hard line break.
      para.forEach((text, index) => {
        if (index) p.appendChild(document.createTextNode(" "));
        inline(p, text.replace(/\\$/, ""), opts);
      });
      root.appendChild(p);
    }
    return root;
  }

  function renderList(parent, lines, i, opts) {
    const first = LIST.exec(lines[i]);
    const indent = first[1].length;
    const ordered = /\d/.test(first[2]);
    const list = el(ordered ? "ol" : "ul");
    if (ordered) list.start = parseInt(first[2], 10) || 1;
    let item = null;
    while (i < lines.length) {
      const line = lines[i];
      const m = LIST.exec(line);
      if (m && m[1].length === indent) {
        item = el("li");
        inline(item, m[3], opts);
        list.appendChild(item);
        i++;
      } else if (m && m[1].length > indent && item) {
        i = renderList(item, lines, i, opts);
      } else if (line.trim() && /^\s+/.test(line) && item && !m) {
        // A continuation line of the current item.
        item.appendChild(document.createTextNode(" "));
        inline(item, line.trim(), opts);
        i++;
      } else {
        break;
      }
    }
    parent.appendChild(list);
    return i;
  }

  window.renderMarkdown = render;
})();
