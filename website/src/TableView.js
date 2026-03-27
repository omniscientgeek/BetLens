import React, { useState, useMemo } from "react";
import { matchesSearch, formatColumnName } from "./dataUtils";

/**
 * Flat table view with sortable columns and search filtering.
 */
export default function TableView({ rows, columns, searchQuery }) {
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState(null); // "asc" | "desc" | null

  const handleSort = (col) => {
    if (sortCol !== col) {
      setSortCol(col);
      setSortDir("asc");
    } else if (sortDir === "asc") {
      setSortDir("desc");
    } else if (sortDir === "desc") {
      setSortCol(null);
      setSortDir(null);
    }
  };

  const sortIndicator = (col) => {
    if (sortCol !== col) return <span className="tv-sort-icon">{"\u2195"}</span>;
    if (sortDir === "asc") return <span className="tv-sort-icon tv-sort-active">{"\u2191"}</span>;
    return <span className="tv-sort-icon tv-sort-active">{"\u2193"}</span>;
  };

  // Filter then sort
  const processedRows = useMemo(() => {
    let result = rows;

    // Search filter
    if (searchQuery) {
      result = result.filter((row) => matchesSearch(row, searchQuery));
    }

    // Sort
    if (sortCol && sortDir) {
      result = [...result].sort((a, b) => {
        const av = a[sortCol];
        const bv = b[sortCol];
        if (av === bv) return 0;
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        if (typeof av === "number" && typeof bv === "number") {
          return sortDir === "asc" ? av - bv : bv - av;
        }
        const cmp = String(av).localeCompare(String(bv));
        return sortDir === "asc" ? cmp : -cmp;
      });
    }

    return result;
  }, [rows, searchQuery, sortCol, sortDir]);

  // Decide which columns to show - prioritize readable columns
  const visibleColumns = useMemo(() => {
    // Put simple fields first, then nested fields
    const simple = columns.filter((c) => !c.includes("."));
    const nested = columns.filter((c) => c.includes("."));
    return [...simple, ...nested];
  }, [columns]);

  // Get a short display label for the column
  const getColumnGroup = (col) => {
    const parts = col.split(".");
    if (parts.length <= 1) return null;
    // For "markets.spread.home_line" return "spread"
    // For "markets.moneyline.home_odds" return "ml"
    if (parts[0] === "markets" && parts.length >= 3) {
      const market = parts[1];
      if (market === "moneyline") return "ml";
      if (market === "spread") return "sprd";
      if (market === "total") return "tot";
      return market;
    }
    return parts[0];
  };

  const highlightCell = (value) => {
    if (!searchQuery || value === null || value === undefined) return String(value ?? "");
    const str = String(value);
    const escaped = searchQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(${escaped})`, "gi");
    const parts = str.split(regex);
    if (parts.length === 1) return str;
    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i} className="tv-match">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  return (
    <div className="tv-container">
      <div className="tv-info">
        {processedRows.length} of {rows.length} rows
        {searchQuery && ` matching "${searchQuery}"`}
      </div>
      <div className="tv-scroll">
        <table className="tv-table">
          <thead>
            <tr>
              {visibleColumns.map((col) => {
                const group = getColumnGroup(col);
                return (
                  <th
                    key={col}
                    className={`tv-th ${group ? `tv-th--${group}` : ""}`}
                    onClick={() => handleSort(col)}
                    title={col}
                  >
                    <div className="tv-th-content">
                      {group && <span className="tv-th-group">{group}</span>}
                      <span className="tv-th-name">{formatColumnName(col)}</span>
                      {sortIndicator(col)}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {processedRows.map((row, idx) => (
              <tr key={idx} className={idx % 2 === 0 ? "tv-tr-even" : "tv-tr-odd"}>
                {visibleColumns.map((col) => (
                  <td key={col} className="tv-td">
                    {highlightCell(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
