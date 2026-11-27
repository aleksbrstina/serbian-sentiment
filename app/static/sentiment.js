/*
 * Browser/Node implementation of the project's text normalization and of
 * TF-IDF + logistic regression inference, driven by the JSON exported by
 * `sentiment.export_static`. It mirrors scikit-learn exactly:
 *   - word analyzer: lower-case, token pattern (?u)\b\w\w+\b, word n-grams
 *   - char_wb analyzer: lower-case, collapse whitespace runs, pad each word with
 *     spaces, character n-grams (a short word is counted once)
 *   - tf -> 1 + ln(tf) if sublinear, times idf, l2-normalized per block
 *   - p(positive) = sigmoid(intercept + sum(block . coef))
 * tests/test_static_demo.py checks parity with Python on the same inputs.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.SentimentModel = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Characters Python's str.isspace() accepts (and the regex \s with re.UNICODE).
  const PY_SPACE = new Set(
    [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000]
      .concat(Array.from({ length: 11 }, (_, i) => 0x2000 + i))
      .map((c) => String.fromCodePoint(c))
  );
  const CONTROL_OR_FORMAT = /[\p{Cc}\p{Cf}]/u;
  const NAMED_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };

  function unescapeHtml(text) {
    return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, body) => {
      if (body[0] === "#") {
        const code = body[1] === "x" || body[1] === "X" ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
        return Number.isFinite(code) && code > 0 && code < 0x110000 ? String.fromCodePoint(code) : match;
      }
      const named = NAMED_ENTITIES[body.toLowerCase()];
      return named === undefined ? match : named;
    });
  }

  function stripHtml(text) {
    text = unescapeHtml(text);
    text = text.replace(/<\s*(br|p|div|li)\b[^<>]*>/gi, " ");
    return text.replace(/<\s*\/?\s*[a-zA-Z][^<>]*>/g, "");
  }

  function removeInvisible(text) {
    let out = "";
    for (const ch of text) {
      if (ch === "\n" || ch === "\t" || ch === " ") out += ch;
      else if (PY_SPACE.has(ch)) out += " ";
      else if (!CONTROL_OR_FORMAT.test(ch)) out += ch;
    }
    return out;
  }

  function isLetter(ch) {
    return /\p{L}/u.test(ch);
  }

  function cyrillicToLatin(text, tables) {
    if (!/\p{Script=Cyrillic}/u.test(text)) return text;
    const chars = Array.from(text);
    const map = tables.cyr_to_lat;
    const digraphs = tables.cyr_upper_digraphs;
    let out = "";
    for (let i = 0; i < chars.length; i++) {
      const ch = chars[i];
      if (digraphs[ch]) {
        let upper = false;
        for (const j of [i + 1, i - 1]) {
          if (j >= 0 && j < chars.length && isLetter(chars[j])) {
            upper = chars[j] === chars[j].toUpperCase() && chars[j] !== chars[j].toLowerCase();
            break;
          }
        }
        out += upper ? digraphs[ch][1] : digraphs[ch][0];
      } else if (map[ch] !== undefined) {
        out += map[ch];
      } else if (map[ch.toLowerCase()] !== undefined) {
        out += map[ch.toLowerCase()].toUpperCase();
      } else {
        out += ch;
      }
    }
    return out;
  }

  function normalizeText(text, tables) {
    text = stripHtml(text || "");
    text = removeInvisible(text);
    text = cyrillicToLatin(text, tables);
    text = text.normalize("NFC");
    text = Array.from(text, (ch) => (tables.diacritic_fixes[ch] !== undefined ? tables.diacritic_fixes[ch] : ch)).join("");
    for (const [src, dst] of Object.entries(tables.punct_fixes)) text = text.split(src).join(dst);
    let collapsed = "";
    let inSpace = false;
    for (const ch of text) {
      if (PY_SPACE.has(ch)) {
        if (!inSpace) collapsed += " ";
        inSpace = true;
      } else {
        collapsed += ch;
        inSpace = false;
      }
    }
    return collapsed.trim();
  }

  // --- analyzers ------------------------------------------------------------------

  function wordTokens(lower) {
    // (?u)\b\w\w+\b == maximal runs of word characters with length >= 2
    return (lower.match(/[\p{L}\p{N}_]+/gu) || []).filter((t) => Array.from(t).length >= 2);
  }

  function wordNgrams(tokens, minN, maxN) {
    if (maxN === 1) return tokens;
    const out = minN === 1 ? tokens.slice() : [];
    for (let n = Math.max(minN, 2); n <= Math.min(maxN, tokens.length); n++) {
      for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
    }
    return out;
  }

  function charWbNgrams(lower, minN, maxN) {
    let text = "";
    let prevSpace = false;
    for (const ch of lower) {
      // sklearn: re.sub(r"\s\s+", " ", text) -- runs of 2+ whitespace become one space
      const space = PY_SPACE.has(ch);
      if (space && prevSpace) {
        text = text.slice(0, -1) + " ";
        continue;
      }
      text += ch;
      prevSpace = space;
    }
    // str.split(): split on runs of Python whitespace
    const words = [];
    let current = "";
    for (const ch of text) {
      if (PY_SPACE.has(ch)) {
        if (current) words.push(current);
        current = "";
      } else {
        current += ch;
      }
    }
    if (current) words.push(current);
    const out = [];
    for (const word of words) {
      const w = Array.from(" " + word + " ");
      for (let n = minN; n <= maxN; n++) {
        let offset = 0;
        out.push(w.slice(offset, offset + n).join(""));
        while (offset + n < w.length) {
          offset += 1;
          out.push(w.slice(offset, offset + n).join(""));
        }
        if (offset === 0) break;
      }
    }
    return out;
  }

  // --- model ----------------------------------------------------------------------

  function prepare(model) {
    for (const block of model.blocks) {
      if (!block.index) {
        block.index = new Map(block.terms.map((t, i) => [t, i]));
      }
    }
    return model;
  }

  function blockVector(block, lower) {
    const [minN, maxN] = block.ngram_range;
    const grams = block.analyzer === "word" ? wordNgrams(wordTokens(lower), minN, maxN) : charWbNgrams(lower, minN, maxN);
    const counts = new Map();
    for (const g of grams) {
      const idx = block.index.get(g);
      if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    const entries = [];
    let norm = 0;
    for (const [idx, tf] of counts) {
      const value = (block.sublinear_tf ? 1 + Math.log(tf) : tf) * block.idf[idx];
      entries.push([idx, value]);
      norm += value * value;
    }
    norm = Math.sqrt(norm) || 1;
    return entries.map(([idx, value]) => [idx, value / norm]);
  }

  /**
   * Predict one (already normalized or raw) text.
   * @returns {{normalized: string, probs: Object<string, number>, label: string,
   *            wordContributions: Map<string, number>}}
   *   wordContributions: TF-IDF value x coefficient per word feature; > 0 pushes
   *   towards the positive class (classes[1]).
   */
  function predict(model, text, options) {
    prepare(model);
    const normalized = options && options.skipNormalization ? text : normalizeText(text, model.normalization);
    const lower = normalized.toLowerCase();
    let z = model.intercept;
    const wordContributions = new Map();
    for (const block of model.blocks) {
      for (const [idx, value] of blockVector(block, lower)) {
        const c = value * block.coef[idx];
        z += c;
        if (block.analyzer === "word") wordContributions.set(block.terms[idx], c);
      }
    }
    const pPos = 1 / (1 + Math.exp(-z));
    const [neg, pos] = model.classes;
    const probs = { [neg]: 1 - pPos, [pos]: pPos };
    // Ties (p = 0.5) go to classes[0], like numpy argmax in predict_proba-based predict().
    return { normalized, probs, label: pPos > 0.5 ? pos : neg, wordContributions };
  }

  return { normalizeText, predict, wordTokens, charWbNgrams };
});
