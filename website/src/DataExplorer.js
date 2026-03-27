import React, { useState, useMemo } from "react";
import TableView from "./TableView";
import PivotView from "./PivotView";
import { findLargestArray, flattenObjects } from "./dataUtils";

const VIEW_MODES = [
  { id: "table", label: "Table", icon: "\uD83D\uDCCA" },
  { id: "pivot", label: "Pivot", icon: "\uD83D\uDD00" },
];

/**
 * Main wrapper for multi-view data exploration.
 * Detects tabular arrays in the data and enables Table/Pivot views.
 */
export default function DataExplorer({ data }) {
  const [viewMode, setViewMode] = useState("table");
  const [searchQuery, setSearchQuery] = useState("");

  // Detect the largest array of objects in the data (e.g. the "odds" array)
  const arrayData = useMemo(() => findLargestArray(data), [data]);

  // Flatten for table view
  const { rows, columns } = useMemo(() => {
    if (!arrayData) return { rows: [], columns: [] };
    return flattenObjects(arrayData.items);
  }, [arrayData]);

  const hasTabularData = arrayData && rows.length > 0;

  if (!hasTabularData) {
    return (
      <div className="de-container">
        <div className="de-empty">No tabular data detected in this file.</div>
      </div>
    );
  }

  return (
    <div className="de-container">
      {/* Toolbar: view mode tabs + search */}
      <div className="de-toolbar">
        <div className="de-view-tabs">
          {VIEW_MODES.map((mode) => (
            <button
              key={mode.id}
              className={`de-tab ${viewMode === mode.id ? "de-tab--active" : ""}`}
              onClick={() => setViewMode(mode.id)}
              title={`Switch to ${mode.label} view`}
            >
              <span className="de-tab-icon">{mode.icon}</span>
              {mode.label}
            </button>
          ))}
        </div>

        <div className="de-search">
          <span className="de-search-icon">{"\uD83D\uDD0D"}</span>
          <input
            type="text"
            className="de-search-input"
            placeholder="Search keys, values..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              className="de-search-clear"
              onClick={() => setSearchQuery("")}
              title="Clear search"
            >
              {"\u2715"}
            </button>
          )}
        </div>
      </div>

      {/* Array detection info */}
      <div className="de-array-info">
        Viewing <strong>{arrayData.key}</strong> — {arrayData.items.length} records,{" "}
        {columns.length} fields
      </div>

      {/* View content */}
      <div className="de-content">
        {viewMode === "table" && (
          <TableView rows={rows} columns={columns} searchQuery={searchQuery} />
        )}
        {viewMode === "pivot" && (
          <PivotView items={arrayData.items} searchQuery={searchQuery} />
        )}
      </div>
    </div>
  );
}
