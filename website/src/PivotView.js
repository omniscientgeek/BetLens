import React, { useState, useMemo } from "react";
import { getNestedValue, extractFieldPaths, matchesSearch, flattenObject } from "./dataUtils";

/**
 * Pivot / cross-tab view: pick a row field, column field, and value field
 * to create a comparison grid. Highlights min/max per row.
 */
export default function PivotView({ items, searchQuery }) {
  // Extract available field paths from the first item
  const fieldPaths = useMemo(() => {
    if (!items || items.length === 0) return [];
    return extractFieldPaths(items[0]);
  }, [items]);

  // Separate string fields (for row/col grouping) from numeric fields (for values)
  const { stringFields, numericFields } = useMemo(() => {
    if (!items || items.length === 0) return { stringFields: [], numericFields: [] };
    const sample = items[0];
    const sf = [];
    const nf = [];
    for (const path of fieldPaths) {
      const val = getNestedValue(sample, path);
      if (typeof val === "number") nf.push(path);
      else if (typeof val === "string") sf.push(path);
    }
    return { stringFields: sf, numericFields: nf };
  }, [items, fieldPaths]);

  // Smart defaults
  const defaultRowField = stringFields.includes("game_id")
    ? "game_id"
    : stringFields.includes("home_team")
    ? "home_team"
    : stringFields[0] || "";
  const defaultColField = stringFields.includes("sportsbook")
    ? "sportsbook"
    : stringFields[1] || "";
  const defaultValueField = numericFields.includes("markets.spread.home_line")
    ? "markets.spread.home_line"
    : numericFields[0] || "";

  const [rowField, setRowField] = useState(defaultRowField);
  const [colField, setColField] = useState(defaultColField);
  const [valueField, setValueField] = useState(defaultValueField);

  // Build pivot data
  const { pivotMap, rowKeys, colKeys } = useMemo(() => {
    const map = {};
    const rSet = new Set();
    const cSet = new Set();

    for (const item of items) {
      const rVal = String(getNestedValue(item, rowField) ?? "—");
      const cVal = String(getNestedValue(item, colField) ?? "—");
      const vVal = getNestedValue(item, valueField);

      rSet.add(rVal);
      cSet.add(cVal);
      if (!map[rVal]) map[rVal] = {};
      map[rVal][cVal] = vVal;
    }

    return {
      pivotMap: map,
      rowKeys: Array.from(rSet).sort(),
      colKeys: Array.from(cSet).sort(),
    };
  }, [items, rowField, colField, valueField]);

  // Compute min/max per row for color coding
  const rowStats = useMemo(() => {
    const stats = {};
    for (const rk of rowKeys) {
      const values = colKeys
        .map((ck) => pivotMap[rk]?.[ck])
        .filter((v) => typeof v === "number");
      if (values.length > 0) {
        stats[rk] = { min: Math.min(...values), max: Math.max(...values) };
      }
    }
    return stats;
  }, [pivotMap, rowKeys, colKeys]);

  // Search filtering on row keys
  const filteredRowKeys = useMemo(() => {
    if (!searchQuery) return rowKeys;
    const q = searchQuery.toLowerCase();
    return rowKeys.filter((rk) => {
      // Match row key itself
      if (rk.toLowerCase().includes(q)) return true;
      // Match any cell value in this row
      for (const ck of colKeys) {
        const v = pivotMap[rk]?.[ck];
        if (v !== null && v !== undefined && String(v).toLowerCase().includes(q)) return true;
      }
      // Also check if the column headers match (show all rows)
      return false;
    });
  }, [rowKeys, colKeys, pivotMap, searchQuery]);

  const getCellClass = (rk, ck) => {
    const val = pivotMap[rk]?.[ck];
    const stats = rowStats[rk];
    if (typeof val !== "number" || !stats || stats.min === stats.max) return "pv-cell";
    if (val === stats.min) return "pv-cell pv-cell--min";
    if (val === stats.max) return "pv-cell pv-cell--max";
    return "pv-cell";
  };

  const formatLabel = (path) => {
    const parts = path.split(".");
    return parts
      .map((p) =>
        p
          .split("_")
          .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
          .join(" ")
      )
      .join(" > ");
  };

  return (
    <div className="pv-container">
      <div className="pv-controls">
        <div className="pv-control">
          <label className="pv-label">Rows</label>
          <select
            className="pv-select"
            value={rowField}
            onChange={(e) => setRowField(e.target.value)}
          >
            {stringFields.map((f) => (
              <option key={f} value={f}>
                {formatLabel(f)}
              </option>
            ))}
          </select>
        </div>
        <div className="pv-control">
          <label className="pv-label">Columns</label>
          <select
            className="pv-select"
            value={colField}
            onChange={(e) => setColField(e.target.value)}
          >
            {stringFields.map((f) => (
              <option key={f} value={f}>
                {formatLabel(f)}
              </option>
            ))}
          </select>
        </div>
        <div className="pv-control">
          <label className="pv-label">Values</label>
          <select
            className="pv-select"
            value={valueField}
            onChange={(e) => setValueField(e.target.value)}
          >
            {numericFields.map((f) => (
              <option key={f} value={f}>
                {formatLabel(f)}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="pv-legend">
        <span className="pv-legend-item">
          <span className="pv-legend-swatch pv-legend-swatch--min" />
          Lowest in row
        </span>
        <span className="pv-legend-item">
          <span className="pv-legend-swatch pv-legend-swatch--max" />
          Highest in row
        </span>
      </div>

      <div className="pv-info">
        {filteredRowKeys.length} of {rowKeys.length} rows
        {searchQuery && ` matching "${searchQuery}"`}
      </div>

      <div className="pv-scroll">
        <table className="pv-table">
          <thead>
            <tr>
              <th className="pv-th pv-th-row">{formatLabel(rowField)}</th>
              {colKeys.map((ck) => (
                <th key={ck} className="pv-th">
                  {ck}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRowKeys.map((rk) => (
              <tr key={rk}>
                <td className="pv-td pv-td-row">{rk}</td>
                {colKeys.map((ck) => {
                  const val = pivotMap[rk]?.[ck];
                  return (
                    <td key={ck} className={getCellClass(rk, ck)}>
                      {val !== null && val !== undefined ? val : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
