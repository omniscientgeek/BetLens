# Sync BetLens & History Pages

Ensures UI changes on the BetLens page (`website/src/App.js` — the `BetLens` component) are mirrored on the History page (`website/src/PastRuns.js` — the `RunDetail` component), and vice versa. Use this skill whenever modifying either page's display of analysis results.

## Trigger

Activate this skill when any change is made to:
- The BetLens page result display (post-pipeline completion UI in `App.js`)
- The History / Past Runs detail view (`RunDetail` in `PastRuns.js`)
- Any shared component used by both: `BriefPanel.js`, `AnalyzeConversation.js`, or `VerificationBadge` (exported from `BriefPanel.js`)

## Architecture Overview

Both pages display the same pipeline result data, but in different contexts:

| Aspect | BetLens (`App.js`) | History (`PastRuns.js` → `RunDetail`) |
|--------|-------------------|---------------------------------------|
| **Data source** | Real-time via WebSocket + `pipelineResults` state | Static fetch from `GET /api/saved-results/{filename}` |
| **Streaming** | Yes — incremental chunks during pipeline | No — fully complete data on load |
| **Data shape** | `pipelineResults.analyze`, `pipelineResults.brief` | `data.pipeline_results.analyze`, `data.pipeline_results.brief` |
| **Shared components** | `AnalyzeConversation`, `BriefPanel`, `VerificationBadge` | Same components, `streaming={false}` |
| **Unique features** | File selector, pipeline stepper, ChatPanel, session persistence | Run list, back navigation, active runs polling |

### Shared Components (changes here automatically affect both pages)
- **`BriefPanel.js`** — Renders the betting analysis brief markdown. Also exports `VerificationBadge`.
- **`AnalyzeConversation.js`** — Displays the AI analysis conversation (system/user/assistant messages + tool calls).
- Both pages pass `streaming={false}` for completed results; BetLens also uses `streaming={true}` during live pipeline.

### Result Display Sections (must stay in sync)

Both pages render these sections for completed results — they must show the same information:

1. **Analyze Conversation** — `<AnalyzeConversation>` component
2. **Audit Analyze card** — Verification card with `<VerificationBadge>`, fix badge, and fix history
3. **Brief Panel** — `<BriefPanel>` component
4. **Audit Brief card** — Same verification card pattern as Audit Analyze

## Sync Checklist

When making a change, follow this checklist:

### 1. Identify the change scope
- [ ] Is this a **shared component** change? (`BriefPanel.js`, `AnalyzeConversation.js`) → Change propagates automatically. Verify both pages still render correctly.
- [ ] Is this a **page-specific layout/structure** change? → Must be manually mirrored to the other page.
- [ ] Is this a **new section or data field** being displayed? → Must be added to both pages.
- [ ] Is this a **CSS class** change? → BetLens uses classes from `App.css`; History uses `.pr-*` prefixed classes in the same `App.css`. Ensure both are updated.

### 2. Mirror structural changes

**If you add/modify a section in BetLens** (the post-completion display in `App.js`):
- Open `PastRuns.js` → `RunDetail` component (around line 206)
- Add the equivalent section using `data.pipeline_results` instead of `pipelineResults`
- Always set `streaming={false}` for shared components on the History page
- Use the `.pr-*` CSS class prefix convention for History-specific styles

**If you add/modify a section in History** (`RunDetail` in `PastRuns.js`):
- Open `App.js` → find the post-completion render block (after `pipelineComplete && pipelineResults`)
- Add the equivalent section using `pipelineResults` state
- Consider whether a streaming version is also needed during live pipeline execution

### 3. Data field mapping

When accessing pipeline result data, use the correct path for each page:

| Data | BetLens (`App.js`) | History (`PastRuns.js`) |
|------|-------------------|------------------------|
| Analyze result | `pipelineResults.analyze` | `pr.analyze` (where `pr = data.pipeline_results`) |
| Analyze verification | `pipelineResults.analyze?.verification` | `pr.analyze?.verification` |
| Brief result | `pipelineResults.brief` | `pr.brief` |
| Brief verification | `pipelineResults.brief?.verification` | `pr.brief?.verification` |
| Brief error | `pipelineResults.brief?.error` | `pr.brief?.error` |
| Fix attempts | `.verification.fix_attempts` | `.verification.fix_attempts` |
| Fix history | `.verification.fix_history` | `.verification.fix_history` |

### 4. Verify consistency

After making changes:
- [ ] Both pages render the same sections for completed results
- [ ] Props passed to shared components match (except `streaming`)
- [ ] New CSS classes are applied to both pages (or shared via component-level styles)
- [ ] Error states are handled in both pages
- [ ] Optional chaining (`?.`) is used consistently — History data may have missing fields from older saved runs

## Common Patterns

### Verification Card Pattern (used for both Audit Analyze and Audit Brief)
Both pages must render this same structure for each audit section:

```jsx
{result?.verification && (
  <div className="verification-card">
    <div className="verification-card-header">
      <span className="verification-card-icon">shield</span>
      <h3>Audit [Phase Name]</h3>
      {result.verification.fix_attempts > 0 && (
        <span className="fix-badge fix-badge--complete">
          {/* fix attempt count + verdict */}
        </span>
      )}
    </div>
    <VerificationBadge verification={result.verification} streaming={false} />
    {result.verification.fix_history?.length > 1 && (
      <details className="fix-history">
        {/* fix history timeline */}
      </details>
    )}
  </div>
)}
```

### Adding a New Display Section
1. Add to shared component if it's reusable, or inline in both pages
2. Add to BetLens post-completion block in `App.js`
3. Add to `RunDetail` in `PastRuns.js`
4. Add any new CSS classes to `App.css`
5. Test with both live pipeline completion and saved run viewing

## Files to Review

Always check these files when making changes:
- `website/src/App.js` — BetLens page (look for post-completion render sections)
- `website/src/PastRuns.js` — History page (`RunDetail` component)
- `website/src/BriefPanel.js` — Shared brief renderer
- `website/src/AnalyzeConversation.js` — Shared conversation display
- `website/src/App.css` — All styles (both pages)
