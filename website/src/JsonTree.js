import React, { useState, useCallback } from "react";

function JsonValue({ value, defaultExpanded = true, depth = 0 }) {
  if (value === null) return <span className="jt-null">null</span>;
  if (value === undefined) return <span className="jt-undefined">undefined</span>;

  const type = typeof value;

  if (type === "boolean")
    return <span className="jt-boolean">{value ? "true" : "false"}</span>;
  if (type === "number")
    return <span className="jt-number">{String(value)}</span>;
  if (type === "string") {
    // Check if it looks like a URL
    if (/^https?:\/\//.test(value)) {
      return (
        <span className="jt-string">
          "
          <a href={value} target="_blank" rel="noopener noreferrer" className="jt-link">
            {value}
          </a>
          "
        </span>
      );
    }
    return <span className="jt-string">"{value}"</span>;
  }

  if (Array.isArray(value)) {
    return (
      <JsonArray items={value} defaultExpanded={defaultExpanded} depth={depth} />
    );
  }

  if (type === "object") {
    return (
      <JsonObject obj={value} defaultExpanded={defaultExpanded} depth={depth} />
    );
  }

  return <span>{String(value)}</span>;
}

function JsonArray({ items, defaultExpanded, depth }) {
  const [expanded, setExpanded] = useState(defaultExpanded && depth < 3);

  if (items.length === 0) return <span className="jt-bracket">{"[]"}</span>;

  if (!expanded) {
    return (
      <span>
        <button
          className="jt-toggle"
          onClick={() => setExpanded(true)}
          title="Expand"
        >
          <span className="jt-arrow">&#9654;</span>
        </button>
        <span className="jt-bracket">{"["}</span>
        <span className="jt-collapsed-hint">
          {items.length} {items.length === 1 ? "item" : "items"}
        </span>
        <span className="jt-bracket">{"]"}</span>
      </span>
    );
  }

  return (
    <span>
      <button
        className="jt-toggle"
        onClick={() => setExpanded(false)}
        title="Collapse"
      >
        <span className="jt-arrow jt-arrow--open">&#9654;</span>
      </button>
      <span className="jt-bracket">{"["}</span>
      <div className="jt-children">
        {items.map((item, idx) => (
          <div className="jt-row" key={idx}>
            <span className="jt-index">{idx}</span>
            <span className="jt-colon">: </span>
            <JsonValue value={item} defaultExpanded={defaultExpanded} depth={depth + 1} />
            {idx < items.length - 1 && <span className="jt-comma">,</span>}
          </div>
        ))}
      </div>
      <span className="jt-bracket">{"]"}</span>
    </span>
  );
}

function JsonObject({ obj, defaultExpanded, depth }) {
  const keys = Object.keys(obj);
  const [expanded, setExpanded] = useState(defaultExpanded && depth < 3);

  if (keys.length === 0) return <span className="jt-bracket">{"{}"}</span>;

  if (!expanded) {
    return (
      <span>
        <button
          className="jt-toggle"
          onClick={() => setExpanded(true)}
          title="Expand"
        >
          <span className="jt-arrow">&#9654;</span>
        </button>
        <span className="jt-bracket">{"{"}</span>
        <span className="jt-collapsed-hint">
          {keys.length} {keys.length === 1 ? "key" : "keys"}
        </span>
        <span className="jt-bracket">{"}"}</span>
      </span>
    );
  }

  return (
    <span>
      <button
        className="jt-toggle"
        onClick={() => setExpanded(false)}
        title="Collapse"
      >
        <span className="jt-arrow jt-arrow--open">&#9654;</span>
      </button>
      <span className="jt-bracket">{"{"}</span>
      <div className="jt-children">
        {keys.map((key, idx) => (
          <div className="jt-row" key={key}>
            <span className="jt-key">"{key}"</span>
            <span className="jt-colon">: </span>
            <JsonValue value={obj[key]} defaultExpanded={defaultExpanded} depth={depth + 1} />
            {idx < keys.length - 1 && <span className="jt-comma">,</span>}
          </div>
        ))}
      </div>
      <span className="jt-bracket">{"}"}</span>
    </span>
  );
}

export default function JsonTree({ data }) {
  const [key, setKey] = useState(0);
  const [isExpanded, setIsExpanded] = useState(true);

  const expandAll = useCallback(() => {
    setIsExpanded(true);
    setKey((k) => k + 1);
  }, []);

  const collapseAll = useCallback(() => {
    setIsExpanded(false);
    setKey((k) => k + 1);
  }, []);

  return (
    <div className="jt-container">
      <div className="jt-toolbar">
        <button className="jt-toolbar-btn" onClick={expandAll}>
          Expand All
        </button>
        <button className="jt-toolbar-btn" onClick={collapseAll}>
          Collapse All
        </button>
      </div>
      <div className="jt-tree">
        <JsonValue key={key} value={data} defaultExpanded={isExpanded} depth={0} />
      </div>
    </div>
  );
}
