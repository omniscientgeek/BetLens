import React, { useState, useMemo, useCallback } from "react";

/**
 * Syntax-highlighted raw JSON view with copy-to-clipboard and search highlighting.
 */
export default function RawJsonView({ data, searchQuery }) {
  const [copied, setCopied] = useState(false);

  const jsonText = useMemo(() => JSON.stringify(data, null, 2), [data]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(jsonText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [jsonText]);

  // Syntax highlight the JSON and optionally mark search matches
  const highlighted = useMemo(() => {
    // Tokenize JSON for syntax coloring
    const colorized = jsonText.replace(
      /("(?:[^"\\]|\\.)*")\s*:/g, // keys
      '<span class="rv-key">$1</span>:'
    ).replace(
      /:\s*("(?:[^"\\]|\\.)*")/g, // string values
      ': <span class="rv-string">$1</span>'
    ).replace(
      /:\s*(-?\d+\.?\d*)/g, // numbers
      ': <span class="rv-number">$1</span>'
    ).replace(
      /:\s*(true|false)/g, // booleans
      ': <span class="rv-boolean">$1</span>'
    ).replace(
      /:\s*(null)/g, // null
      ': <span class="rv-null">$1</span>'
    );

    if (!searchQuery) return colorized;

    // Highlight search matches (escape regex special chars)
    const escaped = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(${escaped})`, "gi");
    return colorized.replace(
      />([^<]*)</g,
      (match, content) => {
        const marked = content.replace(regex, '<mark class="rv-match">$1</mark>');
        return `>${marked}<`;
      }
    );
  }, [jsonText, searchQuery]);

  return (
    <div className="rv-container">
      <div className="rv-toolbar">
        <button className="rv-copy-btn" onClick={handleCopy}>
          {copied ? "\u2713 Copied!" : "Copy JSON"}
        </button>
        {searchQuery && (
          <span className="rv-search-info">
            Highlighting: "{searchQuery}"
          </span>
        )}
      </div>
      <pre
        className="rv-code"
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />
    </div>
  );
}
