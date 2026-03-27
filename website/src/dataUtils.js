/**
 * Shared utility functions for the DataExplorer views.
 */

/**
 * Recursively flatten a nested object into dot-notation keys.
 * e.g. { markets: { spread: { home_line: -5.5 } } }
 *   => { "markets.spread.home_line": -5.5 }
 */
export function flattenObject(obj, prefix = "") {
  const result = {};
  for (const key of Object.keys(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    const val = obj[key];
    if (val !== null && typeof val === "object" && !Array.isArray(val)) {
      Object.assign(result, flattenObject(val, fullKey));
    } else {
      result[fullKey] = val;
    }
  }
  return result;
}

/**
 * Flatten an array of objects, returning { rows, columns }.
 * columns is the union of all keys across rows, in stable order.
 */
export function flattenObjects(arr) {
  const colSet = new Set();
  const rows = arr.map((item) => {
    const flat = flattenObject(item);
    for (const k of Object.keys(flat)) colSet.add(k);
    return flat;
  });
  return { rows, columns: Array.from(colSet) };
}

/**
 * Extract all leaf-level dot-notation field paths from an object.
 * Used for pivot dropdown options.
 */
export function extractFieldPaths(obj) {
  const paths = [];
  function walk(o, prefix) {
    for (const key of Object.keys(o)) {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      const val = o[key];
      if (val !== null && typeof val === "object" && !Array.isArray(val)) {
        walk(val, fullKey);
      } else {
        paths.push(fullKey);
      }
    }
  }
  if (obj && typeof obj === "object") walk(obj, "");
  return paths;
}

/**
 * Walk a JSON tree and return the largest array of objects found,
 * along with its key name. Returns { key, items } or null.
 */
export function findLargestArray(data) {
  let best = null;

  function walk(val, key) {
    if (Array.isArray(val)) {
      if (
        val.length > 0 &&
        typeof val[0] === "object" &&
        val[0] !== null &&
        !Array.isArray(val[0])
      ) {
        if (!best || val.length > best.items.length) {
          best = { key, items: val };
        }
      }
    } else if (val && typeof val === "object") {
      for (const k of Object.keys(val)) {
        walk(val[k], k);
      }
    }
  }

  walk(data, "root");
  return best;
}

/**
 * Check if any value in a flat row matches a search query (case-insensitive).
 */
export function matchesSearch(row, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  for (const val of Object.values(row)) {
    if (val !== null && val !== undefined && String(val).toLowerCase().includes(q)) {
      return true;
    }
  }
  return false;
}

/**
 * Get a dot-notation value from a nested object.
 * e.g. getNestedValue(obj, "markets.spread.home_line")
 */
export function getNestedValue(obj, path) {
  const parts = path.split(".");
  let current = obj;
  for (const part of parts) {
    if (current === null || current === undefined) return undefined;
    current = current[part];
  }
  return current;
}

/**
 * Categorize columns into groups for better table headers.
 * Returns an array of { label, columns } objects.
 */
export function groupColumns(columns) {
  const groups = new Map();
  for (const col of columns) {
    const parts = col.split(".");
    const group = parts.length > 1 ? parts[0] : "";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(col);
  }
  return Array.from(groups.entries()).map(([label, cols]) => ({
    label,
    columns: cols,
  }));
}

/**
 * Pretty-format a column name for display.
 * "markets.spread.home_line" => "Home Line"
 * "game_id" => "Game ID"
 */
export function formatColumnName(col) {
  const parts = col.split(".");
  const leaf = parts[parts.length - 1];
  return leaf
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
