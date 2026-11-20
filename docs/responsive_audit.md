# The interface on every screen — audit and plan (2026-11-20)

USER 2026-11-20: *"hace la pagina responsiva para PC tablet y telefonos, con cortes por resolucion de pantalla,
fijate que los iconos laterales, la barra de herramientas, el asistente, etc nada de la pagina esta hecha de forma
responsiva, muchas cosas no tienen scroll, y quedan ocultas segun la resolucion, quiero que hagas un analisis
completo de toda la paginas y sus modales, funciones, etc, y debe funcionar perfectamente adaptada a cada pantalla
de forma responsiva"*.

**How this was made.** Read-only, on `ui/src` (72 `.tsx`, 20 `.ts`, 31 `.css`, 33,000 lines): the frame read by the
assistant, three parallel reviews for the panels, the windows and the 3D viewer. NOTHING WAS RENDERED: every size
below is computed from the style rules and the tokens (the pod has no browser the assistant may use). "n.v." = not
verified — it needs a real screen. Paths are relative to `ui/src/`.

**The screens it must work on** (CSS pixels): phone portrait 360–430 × 640–930 · phone landscape 640–930 × 360–430 ·
tablet 768–1024 wide, both ways, touch · small laptop 1366 × 768 (his own PC: about 1366 × 600–650 of page once the
browser's bars are taken) and 1280 × 720 · desktop 1920 × 1080 and up. A touch screen has no hover, no right click,
no wheel, no keyboard.

## 1 · The cause, in one paragraph

The page is one fixed frame: a grid of four columns in pixels (icon rail 48 · left panel 280 · viewer `1fr` · right
panel 320) and three rows (header 36 · body · status bar 28), `height: 100vh`, inside `html, body, #root {overflow:
hidden}` (`styles/layout.css:13-26`, `styles/base.css:21-27`, `styles/tokens.css:215-240`). Nothing reflows: THE WHOLE
SOURCE HOLDS ONE SCREEN-SIZE RULE (`features/features.css:1015`, a tick grid of the download window). Since the page
itself cannot scroll, whatever does not fit its box is CUT, silently — by `#root`, by the viewer
(`styles/layout.css:90-96`), by the side panels (`layout/regions.css:151,163`) or by a panel's body
(`components/ui/Panel.css:60-67`, which hides the horizontal overflow). The density switch (compact 28-px controls
with a mouse, "comfortable" 32 px with a finger: `layout/LayoutContext.tsx:37-41`, `styles/tokens.css:282-311`) is the
only adaptation that exists.

## 2 · The frame

| Region | How it is sized | What happens |
|---|---|---|
| The grid | `48 | --left-w | minmax(0,1fr) | --inspector-w`, widths written in px by `LayoutContext` (left 200–640, right 260–720, dock 120–600), stored in the browser and replayed on any screen (`layout/LayoutContext.tsx:34-35,65-77`, `App.tsx:3475-3476`) | viewer width = page − 648 with both panels at their defaults: **0 px up to 648, 120 at 768, 376 at 1024, 632 at 1280, 718 at 1366**. Nothing stops a panel from squeezing the viewer to nothing; the limits ignore the screen (48 + 640 + 720 = 1408 > 1366). Panels never overlay. |
| Height | `.stac-app {height: 100vh}` under `#root {height: 100%}` | on a phone browser `100vh` is taller than what is visible while its bars show: the bottom strip (status bar, the assistant's input, panel footers, the legend) falls outside with no scroll (n.v.). No `dvh` anywhere but the AR surface (`xr/xr.css:9`). |
| Header (36 px) | one flex row, nothing wraps or hides: logo + name, four menus, session ▸ scan selectors, a search field of at least 200 px, assistant, user (`layout/regions.css:8-105`, `layout/AppHeader.tsx:72-105`) | it needs roughly 1,000 px; narrower, the right end — THE USER MENU (sign out, disconnect, administration) and the assistant's button — is pushed out of the page. |
| Icon rail (48 px) | a flex column, top group of 7 (sessions, instances, tickets, scans, BIM, documentation, Gantt) and bottom group of 3 (team, console, settings), `justify-content: space-between`, NO overflow rule (`layout/regions.css:108-144`, `App.tsx:3331-3355`) | 10 buttons of 40 px = **464 px**; the rail has page height − 64. **Under 528 px of page height the icons are cut** and cannot be reached: a landscape phone (296–366 px available), a short window, a zoomed browser. |
| Left panel / right panel | `overflow: hidden`, a 5-px strip to resize it listening to MOUSE events only (`layout/SidePanel.tsx:17`, `Inspector.tsx:20`, `LayoutContext.tsx:144-169`) | cannot be resized by touch; cannot overlay. |
| Bottom dock | height in px, `flex-shrink: 0` (`styles/layout.css:103-106`); its head is tabs + actions that do not shrink (`layout/regions.css:203-220`) | on a landscape phone an open dock leaves the viewer ~66 px; its 600-px limit exceeds the 556 px available at 1366 × 650; with a narrow middle column its last tabs and its close button are clipped. |
| Status bar (28 px) | `white-space: nowrap; overflow: hidden`, every item `flex-shrink: 0` but one (`layout/regions.css:231-256`) | narrow: the right end (points, fps, the console button) is cut. |
| Command palette | 560 × 460 with `max-width: calc(100vw − 32px)` and a max-height (`layout/regions.css:305-328`) | fits every size. |

## 3 · The viewer

**The toolbar** (`components/viewport/FloatingToolbar.tsx:153-211`, `components/viewport/viewport.css:23-54`). 15 icon
buttons always, up to 19 (brush options, camera poses, the comparison's buttons) and a text chip with the deviation
sheet: **541 / 571 / 646 / 676 px** for 15 / 16 / 18 / 19 buttons with a mouse (601–752 px with a finger), about
760–775 px with the chip. It is centred over the viewer; its `overflow-x: auto` can never act (the bar is a centred,
unstretched flex item: it takes its content's width, only its wrapper has the `max-width`), so **narrower than the bar,
the viewer cuts it on BOTH sides — no scroll, no wrap**. And the bar ignores the view cube (a fixed 148 × 148 box at
the top right, `viewport.css:125-149`): under bar + 320 px of viewer it runs over the cube.
- His PC, default panels (viewer 718 px): the bar with camera poses (571) is 86 px inside the cube's box; with the
  deviation sheet (≈ 770) it is cut ≈ 25 px per side. At 1280 the comparison's buttons already push it out.
- A phone with both panels closed (viewer 312–382): 5–8 buttons unreachable — Navigate, Reset view, Grid, Axes,
  Fullscreen.
- Its popovers (brush, display) open under it, `min-width: 360px`, NO max-height and no scroll
  (`viewport.css:56-69`): the brush's is ≈ 420–480 px tall — at 1366 × 650 with the dock open (viewer 356 px high)
  its last rows (merge, clear marks, undo) are off the viewer. The calibration menu is fixed at its button with no
  clamp (`components/ui/Menu.tsx:51-59`).

**What is drawn over the canvas** (all absolute, in px, none aware of the others):

| Overlay | Where | Size | Evidence |
|---|---|---|---|
| View cube | top 12, right 12 | 148 × 148 fixed | `viewport.css:125-149`, `ViewCube.tsx:46` |
| Gravity chip, scale bar, grid text | bottom 12, left 12 | ≈ 410–480 px, wraps | `viewport.css:197-246` |
| Deviation legend | bottom 12, right 12, nowrap | ≈ 455 px | `viewport.css:261-276`, `ui/Legend.css:46-50` |
| Pipeline banner | top 60, centred | `min(100% − 32px, 640px)` | `viewport.css:116-123` |
| Measurement / selection readouts | at the projected point, not clamped | chips | `viewport.css:292-299`, `Viewport.tsx:3912-3934` |
| Ticket and common-point pins | at the projected point | 20 px, text only in `title` | `viewport.css:343-361`, `ViewportHud.tsx:131` |
| Align card | top 108, right 12 | min 200 px | `viewport.css:389-393` |
| Volume card | top 60, centred, one row | ≈ 450–500 px | `viewport.css:395-402`, `Viewport.tsx:5925-5935` |
| Object menu (New ticket) | fixed at the click, no clamp | ≥ 200 px | `App.tsx:3872-3878` |
| BIM element menu | absolute at the click, clipped by the viewer | ≈ 200 × 170 | `Viewport.tsx:5949-6013`, `viewport.css:453-467` |
| Physics panel | left 16, bottom 56, no max-height | 300 × ≈ 330 | `components/components.css:1283-1299` |
| Flythrough | video left half, canvas right half, bar 76 px | fixed halves | `styles/layout.css:109-117` |

Collisions: the legend lies over the scale bar and grid text whenever the viewer is under ≈ 890 px (so on his PC with
default panels); the align card covers the cube's lower 52 px at every size; on a 360-px viewer the volume card is cut
on both sides and the legend loses its left 107 px; the physics panel's title and close button are cut when the viewer
is under ≈ 390 px high (his PC with the dock open). The flythrough's bar height `--sync-bar-h` is defined only for the
compact density (`styles/tokens.css:291`): with a finger (comfortable) its four rules are void and its layout breaks
(`styles/layout.css:114`, `components/components.css:434,478,489`).

**Input.** The camera is three's `OrbitControls` with its defaults (`Viewport.tsx:3804-3813`): one finger rotates, two
pan and pinch (n.v. on a device). Everything the application adds listens to MOUSE events — `click`, `mousedown`,
`mousemove`, `mouseup`, `contextmenu`, and `keydown` on the window (`Viewport.tsx:848-850, 1056-1058, 4544-4559, 4823,
4830, 4888, 4909, 4913`); pointer events only for the physics sphere and the view cube.

| Function | Its only trigger | By touch |
|---|---|---|
| Brush: mark a zone, drop a box | right click (`Viewport.tsx:4392-4447`) | **none** |
| Brush: remove one mark | Shift + right click | none (Clear marks removes all) |
| Object menu → New ticket | right click on the selected object (`Viewport.tsx:4847-4866`) | none in the interface (only the spoken command) |
| BIM element: transparency, hide, isolate | right click (`Viewport.tsx:4867-4886`) | none in the viewer |
| Section box faces | mouse drag on 4-cm handles (`Viewport.tsx:1821, 1863-1958`) | **none** — a finger drag orbits |
| Cancel a pending measurement | Esc or right click | only Clear measurements (all) |
| Close a calibration plane | Enter | a tap within 12 px of the first marker |
| Names of the toolbar's buttons | hover (`ui/Tooltip.tsx:65-69`) — the buttons are icons only | none |
| Snap ring, live distance, brush cursor | hover | none: placed blind |
| Zoom | wheel | pinch |
| Fullscreen | `requestFullscreen()` unguarded (`App.tsx:1884-1887`) | not on an iPhone (n.v.) |

**Size and load.** A `ResizeObserver` follows the container (`Viewport.tsx:4005-4015`) — panels, dock and rotation are
followed. `renderer.setPixelRatio(window.devicePixelRatio)` is UNCAPPED with antialias (`Viewport.tsx:3642-3644`; the
flythrough alone caps it to 1 because the full ratio "saturated the GPU"); the point budget is 10 M by default
(`App.tsx:393`), stored with the session's preferences, not per device; a frame is rendered on every tick; the cube is
a second WebGL context. A phone gets a desktop's load.

## 4 · Panels, lists and pages

Sound vertically: a panel is a flex column whose BODY scrolls (`components/ui/Panel.css`). What breaks is the width —
rows that neither wrap nor shrink, so what gives way is the text the user reads — and everything that needs a mouse.

| Where | Defect | Evidence |
|---|---|---|
| **Assistant** (right panel, its default tab) | input row = microphone + hands-free + text + speaker + send on one line; an `<input>` cannot go under ≈ 170–200 px, so at the default 320 px **Send, then the speaker, slide under the panel's edge** (computed, n.v.). The conversation gets the panel − ≈ 142 px: ≈ 146 px on a 360-px-high screen. The volume form is three number fields a row (≈ 580 px in 277). Tool results only in hover tooltips. On a phone it starts at x = 328: invisible. The on-screen keyboard is not handled (no `visualViewport`). | `components/AssistantPanel.tsx:445-528`, `components/components.css:647-655, 757-761, 811-816`, `components/ui/Field.css:41-51` |
| **Sessions** | a session's second line holds its data + 6–10 icon buttons on one strip: with 10 actions 258 px in 237 → **Delete is cut**, frames / size get 0 px. A scan opens only by double click or Enter; "set reference" is hidden until hover and asks no confirmation. | `features/SessionsPanel.tsx:167-186, 206-208, 257`, `components/ui/Tree.css:172-179` |
| **Instances** | a row = chevron + two 14-px boxes 4 px apart + dot + name + count + two hidden actions: at 280 px the NAME gets 30–65 px (4–9 characters), no tooltip; rows of 24 / 28 px; "remove object" and "delete volume" are invisible until hover, still clickable, with no confirmation; up to 7,932 rows drawn at once. | `features/InstancesPanel.tsx:119-140, 197, 211` |
| Tickets | fits; a long title cannot be read (ellipsis under a tooltip that says something else); creating one needs a right click. | `features/TicketsPanel.tsx:175`, `features/features.css:903-907` |
| Scans | four-column table: at 280 px the date is cut to "2026-08…" (pccr's two scans read alike). | `features/ScansPanel.tsx:51-62`, `components/ui/Table.css:91-100` |
| BIM tree / Report | body scrolls under 169 px of fixed strips; the element's opacity label ("Opacity · name", nowrap) leaves no slider track for a long name; the eye toggle is hover-hidden; the report's element labels are cut with no tooltip. | `components/BIMNavigator.tsx:125, 221-222`, `components/components.css:873-913` |
| Team | fits 280; towards 200 px its Send is cut (the input rule). | `components/TeamPanel.tsx:223-241` |
| Evaluation (dock) | its side column is the RIGHT PANEL'S live width: content = middle column − that − 48 → 0 once the right panel passes ≈ 500 px, and at 768; in a 200-px dock its file tree gets ≈ 45 px. | `features/features.css:766-774, 804-811` |
| Console / Jobs | wrap and scroll. | `features/DockPanels.tsx`, `features/features.css:294-308` |
| Login | a 360-px card (fits a phone's width) of ≈ 390–460 px in `min-height: 100vh` under the no-scroll root: on a landscape phone **Sign in is at or below the fold, unreachable**. | `pages/LoginPage.tsx`, `components/components.css:8-29` |
| Administration | a large dialog whose body scrolls; a user row (avatar + name + 120-px role + 3 icons, no wrap) leaves the name 0 px at 360. Fine from 768. | `pages/AdminPage.tsx:296, 310-329` |

The primitives that impose it on everything: `Tree` — one line a row, actions `opacity: 0` until hover (invisible, live),
activation by double click (`components/ui/Tree.css:154-167`, `Tree.tsx:66-69`); `Table` — cells `nowrap; max-width: 0;
ellipsis`, column widths from the header words alone, so **a column with an empty header (the row's buttons) collapses
and clips them** (`components/ui/Table.css:30-39, 91-100`; n.v.); `Field` — inputs with no `min-width: 0`, sliders with a
4-px track and a 12-px thumb, a segmented control that cannot wrap (`components/ui/Field.css:41-51, 223-304`); `Button` /
`IconButton` — 28 / 32 px, 24 / 28 small, `nowrap`, an icon button's name only in a hover tooltip
(`components/ui/IconButton.tsx:37`); `Tabs` — scroll sideways but push their row when it is a flex parent
(`components/ui/Tabs.css:1-10`).

## 5 · Windows, menus and messages

**The window primitive is sound** (`components/ui/Dialog.css`): a fixed layer with 24 px of padding; the box has its
size by kind (sm 360 × 420, md 440 × 560, lg 640 × 720 — fixed on purpose, USER 2026-10-01: the height never follows
the content) with `max-width / max-height: 100%`; header and footer do not shrink and THE BODY SCROLLS. So an ordinary
window fits any screen. What it costs: 48 px of a phone's width (also in its "full" kind); a FOOTER THAT NEVER WRAPS
and is right-aligned, so when it is wider than the window its LEFT buttons are clipped, unreachable; a non-blocking
window cannot be moved; its close button is 24 / 28 px.

| Window | Defect | Evidence |
|---|---|---|
| **Segmentation manager** (full) | body `overflow: hidden`, a 280-px side column, a footer of one row — a 160-px field + five buttons ≈ 757 px (≈ 800 in Spanish): it fits from ≈ 1,090 px of page; at 1024 "Propagate" loses 60–110 px, at 768 three buttons are off the page. The keyframe's `max-height: calc(90vh − 200px)` ignores the real box: under ≈ 926 px of page height its bottom is cut — ≈ 30 px on his PC. Negative point = right click, pan = middle button or Shift + drag, undo = Ctrl + Z only. | `components/InteractiveSegmentation.tsx:453, 1014, 1213, 1305`, `components/components.css:978-993, 1069-1075, 1193, 1215-1226` |
| **Reference tools** (full) | body `overflow: hidden`, side column fixed at 320 px: no image area on a phone, ≈ 400 px on a tablet upright. On his PC, with an item selected, the list keeps ≈ 110–130 px, and ≈ 0 while its banners show. Pan needs the middle button or Shift + drag. | `features/ReferenceDialog.tsx:378, 668`, `features/features.css:438-453, 474-478, 540-549` |
| **Common points** | a 640-px window fixed 24 px from the right edge that must stay open while the user clicks the model (closing it cancels the mark); the viewer left free: 374 px at 1366 with the left panel open, 32 px at 1024, none on a phone. Its row actions sit in an empty-header column (clipped, n.v.). | `features/CommonPointsDialog.tsx:96, 111-112`, `App.tsx:3891` |
| **Calibration** (lg) | footer of three buttons ≈ 492 px (541 in Spanish): on a phone "Measure on the model" is off the page; its two tables (11 and 8 columns) need ≈ 550 px; the actions column has an empty header (frame, change role, delete clipped at every size, n.v.). The pin that opens it from a distance is 14 px. | `features/CalibrationDialog.tsx:347, 493, 505`, `components/viewport/viewport.css:321-336` |
| Flythrough player | (the token above) + a control row that does not wrap and needs ≈ 470–500 px: narrower, the speed control and Close are cut — the only other exit is Esc; fixed 50 / 50 split, never stacked. | `components/SyncPlayer.tsx:423`, `components/components.css:423-636` |
| Reconstruct | a scan's row does not wrap: ≈ 430 px in 262 on a phone. | `features/PipelineDialog.tsx:39`, `features/features.css:132-139` |
| Resume / Fuse scans | no wrap in the footer: their left button is clipped on a phone (Resume has no close X). | `features/ResumeDialog.tsx:26`, `components/FuseScansModal.tsx:161` |
| Autosegment / Download-Export / GenRecon / Object library / Settings / Background and grid / comparison options / confirmations / New ticket / add member | fit; only the shared defects (targets, footers). | — |
| Video call | fits; on a phone the fixed 160 × 120 self-view covers ≈ 40 % of the stage. | `components/components.css:385-394` |
| Menus (menu bar, selectors, user, calibration, object) | fixed at their anchor, NO clamp and no flip, 200–560 px wide, `max-height` counted from the window, not from the anchor: near the right or bottom edge they leave the screen. | `components/ui/Menu.tsx:52-54`, `components/ui/Menu.css:1-11` |
| Tooltips | hover / focus only — and they are the ONLY name of every icon button. | `components/ui/Tooltip.tsx:65-69` |
| Toasts | 360 px with `max-width: calc(100vw − 32px)`: fit; up to five, errors never expire, close button 18 px. | `components/ui/Toast.css:1-12, 72-82` |

## 6 · What already bites on his own PC (1366 × 768)

1. The toolbar runs into the view cube (86 px) and is cut on both sides with the deviation sheet shown; nothing stops
   a wider panel from cutting it further.
2. The icon rail's 10 icons need 464 px: fine at 100 % in a maximised window, cut in a shorter window or a zoomed page.
3. The assistant's input row at the panel's default width: Send and the speaker under the edge (computed, n.v.).
4. Sessions: a loaded session's last action (Delete) cut; Instances: names of 4–9 characters; Scans: dates cut.
5. With the dock open: the brush's options without their last rows, the physics panel without its title and close.
6. The deviation legend over the scale bar and the grid text.
7. Segmentation manager: the keyframe's bottom ≈ 30 px cut; Reference tools: the list collapses under its form.
8. Common points: a 640-px window over the model it needs clicks on.
9. Row buttons of the Calibration and Common points tables clipped (n.v.); menus near an edge leave the screen.

## 7 · The plan proposed to him (his decisions: the cuts, what each region becomes, the order)

**One set of cuts for the whole page, written once** (a new `styles/responsive.css`, the frame's and the primitives'
rules; each component's own fixes in its own file with the same values) and a small `useScreen()` in `LayoutContext`
for what a style rule cannot do (one overlay at a time, defaults on a small screen, touch gestures).

| Cut | Page | The frame |
|---|---|---|
| **Wide** | ≥ 1280 px | as today — rail, left panel, viewer, right panel side by side — with limits: a panel can never take the viewer under ≈ 560 px, stored widths are clamped to the screen |
| **Medium** | 768–1279 px (tablets, small windows) | the rail stays; the left and right panels OPEN OVER the viewer (one at a time, a tap outside closes them); the header compacts (menus under one button, search as an icon) |
| **Narrow** | under 768 px (phones upright) | the rail becomes a bar at the BOTTOM (its icons scroll sideways); panels, the assistant and every window take the WHOLE screen with a close button; the toolbar is one row that scrolls sideways; the status bar keeps the connection and the running job |
| **Short** | under 520 px of height, any width (phones on their side, squat windows) | the rail and the toolbar scroll; windows take the whole screen; the dock is at most half the height |

**On a touch screen, whatever its size**: targets of at least 40 px; row actions always visible (never hover-only); a
long press = the right click (object menu / New ticket, BIM element menu, the brush's mark); one tap where a double
click is asked today; panel and dock resized with a finger; inputs at 16 px (an iPhone zooms the page under that);
the assistant follows the on-screen keyboard; `100dvh` in place of `100vh`.

**Order — each stage ends with ONE thing for him to look at:**
1. **His PC first: nothing hidden** (§6) — the rail and the toolbar scroll and stop running into the cube, popovers
   and menus stay inside the screen and scroll, the assistant's row, the sessions / instances rows, the legend, the
   dock's and the panels' limits, the header when tight. Style rules almost only.
2. **Tablet + touch basics** — the Medium frame, 40-px targets, visible row actions, long press, touch resizers.
3. **Phone** — the Narrow and Short frames, whole-screen windows, the assistant with the keyboard.
4. **The viewer by touch** — the brush, the section box, measuring without hover, the full-screen tools
   (segmentation manager, reference tools) and — HIS decision — a lighter default on a phone (pixel ratio capped at 2,
   a smaller point budget per device).

**NOT decided, his:** the cut values and what each region becomes; the phone's rendering defaults; whether the
assistant may render the page on the pod to see each size before he does (refused on 2026-11-19: a browser on the pod
is not installed without his word — Playwright is already a dev dependency of the interface, its browser is not
downloaded; a test login would be needed).
