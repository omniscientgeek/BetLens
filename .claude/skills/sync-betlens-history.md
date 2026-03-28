---
name: sync-betlens-history
description: Keeps the BetLens page, History page, and Active Run Detail in sync when UI changes are made to any pipeline result display.
trigger: When modifying App.js (BetLens result display), PastRuns.js (RunDetail/RunList), ActiveRunDetail.js, BriefPanel.js, AnalyzeConversation.js, or VerificationBadge. Use this skill whenever changes are made to how pipeline results are displayed on any page.
---

# Sync BetLens, History & Active Run Pages

Ensures UI changes across all three views that display pipeline results are kept in sync:
- **BetLens** (`website/src/App.js` — the `BetLens` component)
- **History Detail** (`website/src/PastRuns.js` — the `RunDetail` component)
- **Active Run Detail** (`website/src/ActiveRunDetail.js` — the `ActiveRunDetail` component)

Use this skill whenever modifying any page's display of analysis results.

## Trigger

Activate this skill when any change is made to:
- The BetLens page result display (post-pipeline completion UI in `App.js`)
- The History / Past Runs detail view (`RunDetail` in `PastRuns.js`)
- The Active Run detail view (`ActiveRunDetail.js`)
- The History list view (`RunList` / `ActiveRuns` in `PastRuns.js`) or BetLens "Recent Runs" inline list (in `App.js`)
- Any shared component: `BriefPanel.js`, `AnalyzeConversation.js`, or `VerificationBadge` (exported from `BriefPanel.js`)
- Shared API utility: `api.js` (`API_BASE`, `SOCKET_URL`, `fetchWithRetry`)

## Architecture Overview

Three views display pipeline result data in different contexts:

| Aspect | BetLens (`App.js`) | History (`PastRuns.js` → `RunDetail`) | Active Run (`ActiveRunDetail.js`) |
|--------|-------------------|---------------------------------------|-----------------------------------|
| **Data source** | Real-time via WebSocket + `pipelineResults` state | Static fetch from `GET /api/saved-results/{filename}` | Polling `GET /api/active-runs/{filename}` every 2s |
| **Streaming** | Yes — incremental chunks during pipeline | No — fully complete data on load | No — polls for completed phases |
| **Data shape** | `pipelineResults.analyze`, `pipelineResults.brief` | `pr.analyze`, `pr.brief` (where `pr = data.pipeline_results`) | `pr.analyze`, `pr.brief` (where `pr = data.pipeline_results`) |
| **Shared components** | `AnalyzeConversation`, `BriefPanel`, `VerificationBadge` | Same components, `streaming={false}` | Same components, `streaming={false}` |
| **Draft/Verified badges** | Yes — `draft-wrapper`, `draft-badge`, `verified-badge` | Yes — same pattern | **NO — missing** |
| **Brief Revisions** | Yes — shows superseded brief versions | **NO — missing** | **NO — missing** |
| **Unique features** | File selector, pipeline stepper, ChatPanel, session persistence, debug mode, "Recent Runs" inline list, active run inline viewer | Run list with verdict pills, back navigation, active runs polling, `VerdictPill` component | Live progress stepper, phase spinner, completed/error banners |

### Shared Components (changes here automatically affect all three pages)
- **`BriefPanel.js`** — Renders the betting analysis brief markdown. Also exports `VerificationBadge`.
- **`AnalyzeConversation.js`** — Displays the AI analysis conversation (system/user/assistant messages + tool calls).
- **`api.js`** — Exports `API_BASE`, `SOCKET_URL`, `fetchWithRetry`. Used by all pages for API communication.
- All three pages pass `streaming={false}` for completed results; BetLens also uses `streaming={true}` during live pipeline.

### Result Display Sections (must stay in sync)

All three views render these sections for completed pipeline results — they must show the same information:

1. **Draft/Verified Badge Wrapper** — `draft-wrapper` with `draft-badge` or `verified-badge` around Analyze and Brief
2. **Analyze Conversation** — `<AnalyzeConversation>` component
3. **Audit Analyze Timeline** — Interleaved audit blocks + fix blocks with `<VerificationBadge>`, or simple verification card fallback
4. **Brief Revisions** — Superseded brief versions (currently BetLens only)
5. **Brief Panel** — `<BriefPanel>` component
6. **Audit Brief Timeline** — Same pattern as Audit Analyze

### Known Sync Gaps (to fix when touching these areas)

| Feature | BetLens | History `RunDetail` | `ActiveRunDetail` |
|---------|---------|--------------------|--------------------|
| Draft/Verified badge wrappers | ✅ | ✅ | ❌ Missing |
| Brief Revisions section | ✅ | ❌ Missing | ❌ Missing |

### List Views (must stay in sync)

Two places show the run list — they must display the same run metadata:

| Aspect | BetLens "Recent Runs" (`App.js`) | History List (`PastRuns.js` → `RunList`) |
|--------|----------------------------------|------------------------------------------|
| **Shows** | Last 10 runs inline | All runs |
| **Verdict display** | Inline `pr-verdict-pill` spans | `VerdictPill` component |
| **Runtime display** | Not shown | ✅ `formatElapsed` |
| **Active runs** | Separate `bl-active-runs` section | `ActiveRuns` component at top of list |
| **Click action** | Navigates to `/past-runs` with `state.openFile` | Opens `RunDetail` or `ActiveRunDetail` inline |

## Sync Checklist

When making a change, follow this checklist:

### 1. Identify the change scope
- [ ] Is this a **shared component** change? (`BriefPanel.js`, `AnalyzeConversation.js`, `api.js`) → Change propagates automatically. Verify all three pages still render correctly.
- [ ] Is this a **page-specific layout/structure** change? → Must be manually mirrored to the other two pages.
- [ ] Is this a **new section or data field** being displayed? → Must be added to all three pages.
- [ ] Is this a **CSS class** change? → BetLens uses classes from `App.css`; History uses `.pr-*` prefixed classes in the same `App.css`. Ensure all are updated.
- [ ] Is this a **list view** change? → Check both "Recent Runs" in `App.js` and `RunList` in `PastRuns.js`.

### 2. Mirror structural changes

**If you add/modify a section in BetLens** (the post-completion display in `App.js`):
- Open `PastRuns.js` → `RunDetail` component (around line 220)
- Open `ActiveRunDetail.js` → main render (around line 93)
- Add the equivalent section using `pr.analyze` / `pr.brief` instead of `pipelineResults.analyze` / `pipelineResults.brief`
- Always set `streaming={false}` for shared components on History and Active Run pages
- Use the `.pr-*` CSS class prefix convention for History-specific styles

**If you add/modify a section in History** (`RunDetail` in `PastRuns.js`):
- Open `App.js` → find the post-completion render block (after `pipelineComplete && pipelineResults`)
- Open `ActiveRunDetail.js` → completed phase results section
- Add the equivalent section using `pipelineResults` state (App.js) or `pr` (ActiveRunDetail)
- Consider whether a streaming version is also needed during live pipeline execution in `App.js`

**If you add/modify a section in ActiveRunDetail** (`ActiveRunDetail.js`):
- Open `App.js` → post-completion block
- Open `PastRuns.js` → `RunDetail`
- Add the equivalent section in both

### 3. Data field mapping

When accessing pipeline result data, use the correct path for each page:

| Data | BetLens (`App.js`) | History (`PastRuns.js`) | Active Run (`ActiveRunDetail.js`) |
|------|-------------------|------------------------|-----------------------------------|
| Pipeline results object | `pipelineResults` | `pr` (where `const pr = data.pipeline_results \|\| {}`) | `pr` (where `const pr = data.pipeline_results \|\| {}`) |
| Analyze result | `pipelineResults.analyze` | `pr.analyze` | `pr.analyze` |
| Analyze verification | `pipelineResults?.analyze?.verification` | `pr.analyze?.verification` | `pr.analyze?.verification` |
| Brief result | `pipelineResults.brief` | `pr.brief` | `pr.brief` |
| Brief verification | `pipelineResults?.brief?.verification` | `pr.brief?.verification` | `pr.brief?.verification` |
| Brief error | `pipelineResults?.brief?.error` | `pr.brief?.error` | `pr.brief?.error` |
| Fix history | `.verification.fix_history` | `.verification.fix_history` | `.verification.fix_history` |
| Source file | `selectedFile` | `data.source_file \|\| filename` | `data.filename` |
| Run status | `pipelineComplete` / `pipelineError` | Always complete (saved) | `data.status` (`running` / `complete` / `error`) |

### 4. Verify consistency

After making changes:
- [ ] All three pages render the same sections for completed results
- [ ] Props passed to shared components match (except `streaming`)
- [ ] New CSS classes are applied to all pages (or shared via component-level styles)
- [ ] Error states are handled in all three pages
- [ ] Optional chaining (`?.`) is used consistently — History data may have missing fields from older saved runs
- [ ] List views (BetLens "Recent Runs" and History `RunList`) show consistent metadata

## Common Patterns

### Audit Timeline Pattern (used for both Audit Analyze and Audit Brief)
All three views render the same interleaved timeline of audit blocks and fix blocks:

```jsx
{(() => {
  const phaseV = pr.analyze?.verification;  // or pr.brief?.verification
  const history = phaseV?.fix_history || [];
  const auditEntries = history.filter((h) => h.type !== "fix");
  const fixEntries = history.filter((h) => h.type === "fix");
  const hasTimeline = auditEntries.length > 0;

  if (hasTimeline) {
    const timelineItems = history.map((h, idx) => {
      if (h.type === "fix") {
        return (
          <div key={`fix-${idx}`} className="fix-block">
            <div className="fix-block-header">
              <span className="fix-block-indicator">{"\u{1F527}"}</span>
              <span className="fix-block-title">Fix #{h.attempt}</span>
              {h.ai_meta && (
                <span className="fix-block-meta">
                  {h.ai_meta.provider} / {h.ai_meta.model}
                  {h.ai_meta.elapsed_seconds != null && ` · ${h.ai_meta.elapsed_seconds.toFixed(1)}s`}
                </span>
              )}
            </div>
            {h.conversation && (
              <AnalyzeConversation
                analyzeResult={{ conversation: h.conversation, ai_meta: h.ai_meta }}
                streaming={false}
                defaultExpanded={false}
                title={`Fix #${h.attempt} Conversation`}
              />
            )}
          </div>
        );
      }
      // Audit entry
      const isLatest = idx === history.length - 1 ||
        history.slice(idx + 1).every((e) => e.type === "fix");
      const isPassed = h.verdict === "pass";
      const issueCount = Object.values(h.verification?.agents || {}).reduce(
        (sum, a) => sum + (a.issues?.length || 0), 0
      );
      return (
        <div key={`audit-${idx}`} className={`audit-block audit-block--${h.verdict} ${isLatest ? "audit-block--latest" : "audit-block--historical"}`}>
          <div className="audit-block-header">
            <span className={`audit-block-indicator audit-block-indicator--${h.verdict}`}>
              {isPassed ? "✅" : h.verdict === "warn" ? "⚠️" : "❌"}
            </span>
            <span className="audit-block-title">
              {h.attempt === 0 ? "Initial Audit" : `Re-audit after Fix #${h.attempt}`}
            </span>
            <span className={`audit-block-verdict verdict--${h.verdict}`}>
              {h.verdict.toUpperCase()}
            </span>
            {issueCount > 0 && (
              <span className="audit-block-issues">
                {issueCount} issue{issueCount !== 1 ? "s" : ""}
              </span>
            )}
            {isLatest && <span className="audit-block-latest-tag">Latest</span>}
          </div>
          <VerificationBadge verification={h.verification} />
        </div>
      );
    });

    return (
      <div className="audit-timeline">
        <div className="audit-timeline-header">
          <span className="audit-timeline-icon">{"\u{1F6E1}"}</span>
          <h3>Audit [Phase] Timeline</h3>
          <span className="audit-timeline-count">
            {auditEntries.length} audit{auditEntries.length !== 1 ? "s" : ""}
            {fixEntries.length > 0 && `, ${fixEntries.length} fix${fixEntries.length !== 1 ? "es" : ""}`}
          </span>
        </div>
        {timelineItems}
      </div>
    );
  }

  // Fallback: simple verification card (no timeline)
  if (phaseV) {
    return (
      <div className="verification-card">
        <div className="verification-card-header">
          <span className="verification-card-icon">{"\u{1F6E1}"}</span>
          <h3>Audit [Phase]</h3>
        </div>
        <VerificationBadge verification={phaseV} streaming={false} />
      </div>
    );
  }
  return null;
})()}
```

### Draft/Verified Badge Wrapper Pattern
Used to wrap Analyze and Brief sections (currently in BetLens + History, missing from ActiveRunDetail):

```jsx
<div className={`draft-wrapper ${isDraft ? "draft-wrapper--draft" : hasVerification ? "draft-wrapper--verified" : ""}`}>
  {isDraft && (
    <div className="draft-badge">
      <span className="draft-badge-icon">{"\u26A0\uFE0F"}</span>
      <span>DRAFT — [Section] did not pass all audits</span>
    </div>
  )}
  {!isDraft && hasVerification && (
    <div className="verified-badge">
      <span className="verified-badge-icon">{"\u2705"}</span>
      <span>VERIFIED — [Section] passed all audits</span>
    </div>
  )}
  {/* Section content (AnalyzeConversation or BriefPanel) */}
</div>
```

### Adding a New Display Section
1. Add to shared component if it's reusable, or inline in all three pages
2. Add to BetLens post-completion block in `App.js`
3. Add to `RunDetail` in `PastRuns.js`
4. Add to `ActiveRunDetail.js` completed phase results section
5. Add any new CSS classes to `App.css`
6. Test with: live pipeline completion, saved run viewing, and active run polling

## Files to Review

Always check these files when making changes:
- `website/src/App.js` — BetLens page (post-completion render + "Recent Runs" inline list + streaming views)
- `website/src/PastRuns.js` — History page (`RunList`, `ActiveRuns`, `RunDetail`, `VerdictPill`)
- `website/src/ActiveRunDetail.js` — Active run detail view (polls for live progress, renders completed phases)
- `website/src/BriefPanel.js` — Shared brief renderer + exports `VerificationBadge`
- `website/src/AnalyzeConversation.js` — Shared conversation display
- `website/src/api.js` — Shared API utilities (`API_BASE`, `SOCKET_URL`, `fetchWithRetry`)
- `website/src/App.css` — All styles (all pages)

## Self-Maintenance — Keep This Skill Up To Date

**IMPORTANT:** This skill document must stay accurate as the codebase evolves. Whenever this skill is triggered, perform these checks and update this file if anything has drifted:

### On every use of this skill:
1. **Scan imports** — Check the `import` statements at the top of `App.js`, `PastRuns.js`, and `ActiveRunDetail.js`. If any new component is imported that displays pipeline result data (shared or page-specific), add it to the "Files to Review" list and the architecture table above.
2. **Check for new files** — If a new `.js` file has been created in `website/src/` that is imported by any of the tracked pages, add it to this skill document with a description of its role.
3. **Check for removed files** — If a tracked file no longer exists or is no longer imported, remove it from this document.
4. **Verify architecture table** — Confirm the columns and rows still match reality. Add/remove columns if views were added or removed.
5. **Update sync gaps table** — If a known sync gap has been fixed, mark it ✅. If a new gap is found, add it.
6. **Update data field mapping** — If new data fields are being accessed in any view, add them to the mapping table.
7. **Update trigger list** — If the set of files that should trigger this skill has changed, update both the frontmatter `trigger:` field and the "Trigger" section.

### When creating a new shared component:
- Add it to the "Shared Components" section
- Add it to the "Files to Review" list
- Add it to the frontmatter `trigger:` field
- Document its props and which pages use it

### When creating a new page/view that displays pipeline results:
- Add a new column to the architecture table
- Add it to the "Result Display Sections" sync list
- Add it to the data field mapping table
- Add it to the "Mirror structural changes" instructions
- Add it to the "Adding a New Display Section" checklist
- Update the frontmatter `trigger:` and `description:` fields
