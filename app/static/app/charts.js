// Two small SVG charts for the Insights view, built with DOM APIs.
//
// Mark specs follow the dataviz method: bars <= 24px thick with a 4px rounded
// data end and a square baseline, hairline recessive gridlines, one axis,
// selective direct labels in text colors (never the series color), and a
// hover tooltip per mark. Colors come from CSS custom properties so light and
// dark are each chosen, not flipped.
"use strict";

(function () {
  const NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs, text) {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
    if (text != null) node.textContent = text;
    return node;
  }

  // -- tooltip --------------------------------------------------------------
  const tip = () => document.getElementById("tooltip");

  function showTip(event, lines) {
    const t = tip();
    t.replaceChildren();
    lines.forEach((line, index) => {
      const div = document.createElement("div");
      if (index === 0) {
        const b = document.createElement("b");
        b.textContent = line;
        div.appendChild(b);
      } else {
        div.textContent = line;
      }
      t.appendChild(div);
    });
    t.hidden = false;
    const pad = 12;
    const rect = t.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
    t.style.left = Math.max(8, x) + "px";
    t.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip().hidden = true; }

  function bindTip(node, lines) {
    node.addEventListener("mousemove", (e) => showTip(e, lines));
    node.addEventListener("mouseleave", hideTip);
    node.addEventListener("focus", (e) => {
      const r = node.getBoundingClientRect();
      showTip({ clientX: r.right, clientY: r.top }, lines);
    });
    node.addEventListener("blur", hideTip);
    node.setAttribute("tabindex", "0");
    node.setAttribute("aria-label", lines.join(". "));
  }

  // A bar whose data end is rounded (4px) and whose baseline end is square.
  function hBarPath(x0, y, w, h, r) {
    r = Math.min(r, w, h / 2);
    if (w <= 0) return "";
    return `M${x0},${y} H${x0 + w - r} Q${x0 + w},${y} ${x0 + w},${y + r} V${y + h - r} ` +
           `Q${x0 + w},${y + h} ${x0 + w - r},${y + h} H${x0} Z`;
  }
  function vBarPath(x, yBase, w, h, r) {
    r = Math.min(r, h, w / 2);
    if (h <= 0) return "";
    const top = yBase - h;
    return `M${x},${yBase} V${top + r} Q${x},${top} ${x + r},${top} H${x + w - r} ` +
           `Q${x + w},${top} ${x + w},${top + r} V${yBase} Z`;
  }

  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  // -- horizontal bars: one series, ranked ---------------------------------
  // rows: [{label, value, tip: [str], valueLabel}]
  function hbar(container, rows, opts) {
    opts = opts || {};
    const width = Math.max(320, container.clientWidth || 640);
    const labelW = Math.min(240, Math.round(width * 0.36));
    const valueW = 56;
    const band = 30, thick = 18, top = 6, axisH = 20;
    const height = top + rows.length * band + axisH;
    const plotW = width - labelW - valueW;
    const max = opts.max || niceMax(Math.max(...rows.map((r) => r.value || 0)));
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img",
                               "aria-label": opts.title || "bar chart" });

    const ticks = 4;
    for (let k = 0; k <= ticks; k++) {
      const x = labelW + (plotW * k) / ticks;
      svg.appendChild(svgEl("line", { x1: x, x2: x, y1: top - 2, y2: top + rows.length * band,
                                      class: k === 0 ? "axis" : "gridline" }));
      svg.appendChild(svgEl("text", { x, y: height - 4, "text-anchor": "middle" },
                            (opts.tickFormat || String)((max * k) / ticks)));
    }

    rows.forEach((row, index) => {
      const y = top + index * band + (band - thick) / 2;
      const label = row.label.length > 34 ? row.label.slice(0, 33) + "…" : row.label;
      svg.appendChild(svgEl("text", { x: labelW - 8, y: y + thick / 2 + 4, "text-anchor": "end",
                                      class: "label" }, label));
      const w = Math.max(0, (plotW * (row.value || 0)) / max);
      svg.appendChild(svgEl("path", { d: hBarPath(labelW, y, w, thick, 4),
                                      fill: opts.color || "var(--series-after)" }));
      svg.appendChild(svgEl("text", { x: labelW + w + 6, y: y + thick / 2 + 4, class: "label" },
                            row.valueLabel != null ? row.valueLabel : String(row.value)));
      const hit = svgEl("rect", { x: 0, y: top + index * band, width, height: band, class: "hit" });
      bindTip(hit, row.tip || [row.label, String(row.value)]);
      svg.appendChild(hit);
    });

    const wrap = document.createElement("div");
    wrap.className = "chart";
    wrap.appendChild(svg);
    container.appendChild(wrap);
  }

  // -- grouped columns: before vs after, with 95% CI whiskers --------------
  // groups: [{label, before: {value, n, successes, ci95}, after: {...}}]
  function beforeAfter(container, groups, opts) {
    opts = opts || {};
    const series = [
      { key: "before", name: "Before", color: "var(--series-before)" },
      { key: "after", name: "After", color: "var(--series-after)" },
    ];
    const legend = document.createElement("div");
    legend.className = "legend";
    for (const s of series) {
      const item = document.createElement("span");
      const sw = document.createElement("span");
      sw.className = "sw";
      sw.style.background = s.color;
      item.append(sw, document.createTextNode(s.name));
      legend.appendChild(item);
    }
    const ci = document.createElement("span");
    ci.textContent = "whiskers: 95% CI";
    legend.appendChild(ci);
    container.appendChild(legend);

    const width = Math.max(320, container.clientWidth || 640);
    const height = 230, top = 22, bottom = 40, left = 40;
    const plotH = height - top - bottom;
    const plotW = width - left - 8;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img",
                               "aria-label": opts.title || "before and after" });
    const yOf = (v) => top + plotH * (1 - v);
    for (let k = 0; k <= 4; k++) {
      const v = k / 4;
      svg.appendChild(svgEl("line", { x1: left, x2: width - 8, y1: yOf(v), y2: yOf(v),
                                      class: k === 0 ? "axis" : "gridline" }));
      svg.appendChild(svgEl("text", { x: left - 6, y: yOf(v) + 4, "text-anchor": "end" },
                            Math.round(v * 100) + "%"));
    }
    const slot = plotW / groups.length;
    const thick = Math.min(24, slot / 3.2);
    const gap = 2;
    groups.forEach((group, gi) => {
      const cx = left + slot * gi + slot / 2;
      // Narrow screens: shrink category labels rather than let them collide.
      const size = slot < 95 ? 10 : 11;
      svg.appendChild(svgEl("text", { x: cx, y: height - bottom + 16, "text-anchor": "middle",
                                      class: "label", "font-size": size,
                                      style: `font-size:${size}px` }, group.label));
      if (group.note) {
        svg.appendChild(svgEl("text", { x: cx, y: height - bottom + 30, "text-anchor": "middle" },
                              group.note));
      }
      series.forEach((s, si) => {
        const rate = group[s.key];
        const x = cx + (si === 0 ? -thick - gap / 2 : gap / 2);
        if (!rate || typeof rate.value !== "number") return;
        const h = plotH * Math.max(0, Math.min(1, rate.value));
        svg.appendChild(svgEl("path", { d: vBarPath(x, yOf(0), thick, h, 4), fill: s.color }));
        if (Array.isArray(rate.ci95) && rate.ci95.length === 2) {
          const wx = x + thick / 2;
          const [lo, hi] = rate.ci95;
          svg.appendChild(svgEl("line", { x1: wx, x2: wx, y1: yOf(lo), y2: yOf(hi), class: "whisker" }));
          for (const v of [lo, hi]) {
            svg.appendChild(svgEl("line", { x1: wx - 4, x2: wx + 4, y1: yOf(v), y2: yOf(v),
                                            class: "whisker" }));
          }
        }
        // Direct label on the "after" cap only: the story is the change.
        if (s.key === "after") {
          const hi = Array.isArray(rate.ci95) ? rate.ci95[1] : rate.value;
          svg.appendChild(svgEl("text", { x: x + thick / 2, y: yOf(Math.max(hi, rate.value)) - 5,
                                          "text-anchor": "middle", class: "label" },
                                Math.round(rate.value * 100) + "%"));
        }
        const hit = svgEl("rect", { x: x - 3, y: top, width: thick + 6, height: plotH, class: "hit" });
        bindTip(hit, [`${group.label} — ${s.name.toLowerCase()}`, ...window.vdRateLines(rate)]);
        svg.appendChild(hit);
      });
    });
    const wrap = document.createElement("div");
    wrap.className = "chart";
    wrap.appendChild(svg);
    container.appendChild(wrap);
  }

  window.vdCharts = { hbar, beforeAfter, bindTip, hideTip };
})();
