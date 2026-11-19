// STAC-Builder — the interface's side of THE DATABASE OF COMMANDS (server/commands).
//
// USER 2026-11-14: "el usuario debe poder hablar con stac build y stac build operar sobre todo el
// sistema … preparado para todas las que tiene ahora y en un futuro". A command of the interface
// is DECLARED in the server's catalogue (what it is, its arguments, how people ask for it, what
// STAC says when it is done — in es / en / fr) and EXECUTED here: the chat's answer carries the
// commands the model chose (`ui_actions`), the interface runs each one through this registry and
// words the outcome with the texts the server sent.
//
// The executors are typed against catalog.generated.ts: a command of the catalogue with no
// executor does not compile (see `AppExecutors` in App.tsx), nor does one that is not declared.
import { UI_CONFIRM_IDS } from './catalog.generated'
import type { UiCommandArgs, UiCommandId, UiConfirmId } from './catalog.generated'

/** why a command did not run — worded by the server in the conversation's language (answers.ui_reasons) */
export type Reason = 'no_session' | 'not_connected' | 'server_down' | 'not_found' | 'ambiguous' | 'nothing_to_do' | 'needs_gesture' | 'unavailable' | 'not_wired' | 'failed' | 'one_confirmation' | 'needs_confirmation' | 'window_closed' | 'needs_selection' | 'not_allowed'

export type Outcome =
  /** `detail`: what the command reports (the session it opened, a list); `state`: for an on / off
   *  command, what it left (false = the "off" wording) */
  | { ok: true; detail?: string; state?: boolean }
  | { ok: false; reason: Reason; detail?: string }

export type Executor<K extends UiCommandId = UiCommandId> = (args: UiCommandArgs[K]) => Outcome | Promise<Outcome>
/** THE SPOKEN CONFIRMATION (USER 2026-11-17: "confirmacion hablada con lo que stac nos dice que va
 *  a hacer, STAC, SI CONFIRMO" — server/commands/confirm.py). A command the catalogue declares
 *  `confirm: true` (it launches GPU work or deletes data) has TWO steps: `prepare` changes
 *  NOTHING — it checks the command can run and reports as `detail` exactly what it would act on
 *  (which scan, which object), which STAC says; `run` does it, and is called only after the
 *  user's next sentence confirmed. */
export interface ConfirmExecutor<K extends UiCommandId = UiCommandId> { prepare: Executor<K>; run: Executor<K> }
export type Executors<K extends UiCommandId = UiCommandId> = { [I in K]: I extends UiConfirmId ? ConfirmExecutor<I> : Executor<I> }

/** the commands the Assistant panel owns (the user's name, STAC's own voice); every other one is the app's */
export type PanelCommandId = 'talk.set_user_name' | 'talk.forget_user_name' | 'talk.spoken_answers' | 'talk.hands_free'

/** the commands the 3D viewer owns (its own tools' state: gizmos, the measurement in progress, the Align tool) */
export type ViewportCommandId = 'volumes.set_gizmo_mode' | 'volumes.set_solid' | 'measure.finish_plane' | 'measure.cancel_pending'
  | 'viewer.align_set_gizmo_mode' | 'viewer.align_save' | 'viewer.fly_to_camera_pose'
/** … and the flythrough's player, while it is open */
export type FlythroughCommandId = 'flythrough.play_pause' | 'flythrough.step_frame' | 'flythrough.seek' | 'flythrough.set_speed'

/** … and the Calibration window, while it is open */
export type CalibrationCommandId = 'calibration.save_distance' | 'calibration.save_plane' | 'calibration.toggle_role'
  | 'calibration.delete_measurement' | 'calibration.delete_plane' | 'calibration.set_mode' | 'calibration.apply'

/** … the BIM model's tree, while its panel is open */
export type BimCommandId = 'bim.load_ifc' | 'bim.delete_ifc' | 'bim.set_element_visible' | 'bim.select_element' | 'bim.set_element_opacity'
  | 'bim.filter' | 'bim.ctx_set_transparency' | 'bim.ctx_hide' | 'bim.ctx_isolate' | 'bim.ctx_show_all'

/** … the certificate panel, the Evaluation tab and the Report tab of the dock, while each is open */
export type EvaluationCommandId = 'evaluation.refresh_runs' | 'evaluation.select_run' | 'evaluation.open_file' | 'evaluation.set_json_view'
export type ReportCommandId = 'reports.run_deviation'

/** … the Team panel (a call is started from its members) and the video call while there is one */
export type TeamCommandId = 'team.select' | 'team.set_tab' | 'team.send_message' | 'team.open_add_member' | 'team.add_member' | 'team.remove_member' | 'call.start'
export type CallCommandId = 'call.accept' | 'call.decline' | 'call.end' | 'call.set_mute' | 'call.set_camera'

/** … and the Administration window, while it is open */
export type AdminCommandId = 'admin.set_tab' | 'admin.create_user' | 'admin.set_user_role' | 'admin.toggle_user_active' | 'admin.edit_user'
  | 'admin.delete_user' | 'admin.create_team' | 'admin.delete_team' | 'admin.change_team_manager' | 'admin.add_team_member'
  | 'admin.remove_team_member' | 'admin.assign_session' | 'admin.unassign_session' | 'admin.create_team_session'

/** … the reference window (the references marked on the keyframe images), while it is open */
export type ReferenceCommandId = 'references.set_mode' | 'references.go_to_keyframe' | 'references.zoom' | 'references.place_point'
  | 'references.close_polygon' | 'references.undo_vertex' | 'references.cancel_pending' | 'references.select_item'
  | 'references.arm_add_observation' | 'references.edit_distance' | 'references.edit_plane' | 'references.remove_observation'
  | 'references.remove_polygon' | 'references.delete_item' | 'references.save' | 'references.close'

/** … the Segmentation manager, while it is open, and the Fuse window */
export type ManagerCommandId = 'segmentation.close_manager' | 'segmentation.manager_set_mode' | 'segmentation.manager_go_to_frame'
  | 'segmentation.manager_zoom' | 'segmentation.manager_mark_frame_and_next' | 'segmentation.manager_toggle_frame_selection'
  | 'segmentation.manager_add_point' | 'segmentation.manager_undo_point' | 'segmentation.manager_clear_object' | 'segmentation.manager_clear_all'
  | 'segmentation.manager_text_prompt' | 'segmentation.manager_auto' | 'segmentation.manager_evaluate' | 'segmentation.manager_add_object'
  | 'segmentation.manager_set_label' | 'segmentation.manager_propagate' | 'segmentation.manager_select_instance'
  | 'segmentation.manager_rename_instance' | 'segmentation.manager_toggle_exclude' | 'segmentation.manager_delete_instance'
export type FuseCommandId = 'scans.fuse'

/** … and the Export window */
export type ExportCommandId = 'export.run'

/** one command of an answer, as the server sends it */
export interface UiAction {
  id: string
  args: Record<string, unknown>
  kind: 'action' | 'query' | 'talk'
  confirm: boolean
  /** the server released it: the user's sentence was the confirmation */
  confirmed?: boolean
  /** what STAC says it is going to do ({detail}: what `prepare` reported) */
  ask?: string
  title: string
  done: string
  done_off: string
}

/** the texts an answer carries for the interface to word its commands with */
export interface AnswerTexts {
  done_prefix: string
  name_first?: boolean
  reasons: Record<string, string>
  /** a sentence STAC adds after the commands' outcomes (its name question, asked once more) */
  tail?: string
  /** how a held command is confirmed ("say: STAC, yes, I confirm") */
  confirm_how?: string
}

/** a command STAC is holding until the user's next sentence confirms it — sent back with it */
export interface PendingCommand { id: string; args: Record<string, unknown>; detail: string }

type Entry = ((args: never) => Outcome | Promise<Outcome>) | { prepare: (args: never) => Outcome | Promise<Outcome>; run: (args: never) => Outcome | Promise<Outcome> }
const registry = new Map<string, Entry>()
const CONFIRM = new Set<string>(UI_CONFIRM_IDS)

/** Register executors (a component registers the commands it owns). Returns the un-register. */
export function registerCommands<K extends UiCommandId>(executors: Executors<K>): () => void {
  const ids = Object.keys(executors) as K[]
  for (const id of ids) registry.set(id, executors[id] as unknown as Entry)
  return () => { for (const id of ids) if (registry.get(id) === (executors[id] as unknown)) registry.delete(id) }
}

export function isRegistered(id: string): boolean { return registry.has(id) }

// THE WINDOW A COMMAND LIVES IN: the commands of a panel or a dialog are registered while it is
// open. Asked for with it closed, STAC opens it first — the app says how, per area of the
// catalogue (the id's first part) — and the command runs once the window has registered it.
// An opener returns false when it cannot open now (no session, a window that needs a pick).
const openers = new Map<string, () => boolean | void>()
export function registerOpeners(byArea: Record<string, () => boolean | void>): () => void {
  for (const [area, fn] of Object.entries(byArea)) openers.set(area, fn)
  return () => { for (const [area, fn] of Object.entries(byArea)) if (openers.get(area) === fn) openers.delete(area) }
}
const OPEN_WAIT_MS = 2000
async function opened(id: string): Promise<Entry | undefined> {
  const open = openers.get(id.split('.')[0])
  if (!open || open() === false) return undefined
  const end = Date.now() + OPEN_WAIT_MS
  while (!registry.has(id) && Date.now() < end) await new Promise(r => setTimeout(r, 40))
  return registry.get(id)
}

/** Run one command. Never throws: a failure is an outcome. */
export async function runCommand(id: string, args: Record<string, unknown>, step: 'run' | 'prepare' | 'confirmed' = 'run'): Promise<Outcome> {
  const ex = registry.get(id) ?? await opened(id)
  // (every command of the catalogue has an executor — the compiler holds it; one that is not
  // registered NOW belongs to a window that is not open: the flythrough, a dialog)
  if (!ex) return { ok: false, reason: 'window_closed' }
  try {
    if (typeof ex === 'function') {
      // (the catalogue is the authority: a command it declares with confirm never runs unconfirmed)
      if (CONFIRM.has(id) || step !== 'run') return { ok: false, reason: 'not_wired' }
      return await ex((args ?? {}) as never)
    }
    if (step === 'run') return { ok: false, reason: 'needs_confirmation' }
    return await (step === 'prepare' ? ex.prepare : ex.run)((args ?? {}) as never)
  } catch (e) {
    return { ok: false, reason: 'failed', detail: e instanceof Error ? e.message : String(e) }
  }
}

function fill(template: string, values: Record<string, unknown>): string {
  return template.replace(/\{([a-z_]+)\}/g, (_m, key: string) => {
    const v = values[key]
    if (v == null) return ''
    return typeof v === 'number' ? String(Math.round(v * 100) / 100) : String(v)
  }).replace(/\s+([.:,;])/g, '$1').replace(/\s{2,}/g, ' ').trim()
}

/** What STAC says of one command's outcome, from the server's texts. */
export function wordOutcome(action: UiAction, outcome: Outcome, texts: AnswerTexts): string {
  if (!outcome.ok) {
    const tpl = texts.reasons[outcome.reason] ?? texts.reasons.failed ?? ''
    return fill(tpl, { detail: outcome.detail ?? action.title })
  }
  const off = outcome.state === false || (outcome.state === undefined && action.args.on === false)
  return fill(off && action.done_off ? action.done_off : action.done, { ...action.args, detail: outcome.detail ?? '' })
}

/** Run an answer's commands in order and word the whole outcome: the server's own sentence (its
 *  tools' results) first, then each command's; a turn in which STAC only acted, and all of it
 *  worked, is confirmed by the user's name. */
export async function runActions(actions: UiAction[], serverAnswer: string, texts: AnswerTexts, actedOnly: boolean): Promise<{ text: string; outcomes: Outcome[]; pending?: PendingCommand }> {
  const outcomes: Outcome[] = []
  const said: string[] = []
  let pending: PendingCommand | undefined
  let question = ''
  if (serverAnswer.trim()) said.push(serverAnswer.trim())
  for (const action of actions) {
    if (action.confirm && !action.confirmed) {
      // HELD FOR CONFIRMATION: nothing is changed — STAC says what it is going to do and waits
      if (pending) { said.push(fill(texts.reasons.one_confirmation ?? '', { detail: action.title })); outcomes.push({ ok: false, reason: 'one_confirmation' }); continue }
      const ready = await runCommand(action.id, action.args, 'prepare')
      if (!ready.ok) { outcomes.push(ready); said.push(wordOutcome(action, ready, texts)); continue }
      pending = { id: action.id, args: action.args, detail: ready.detail ?? '' }
      question = `${fill(action.ask ?? action.title, { ...action.args, detail: ready.detail ?? '' })} ${texts.confirm_how ?? ''}`.trim()
      continue
    }
    const outcome = await runCommand(action.id, action.args, action.confirm ? 'confirmed' : 'run')
    outcomes.push(outcome)
    const sentence = wordOutcome(action, outcome, texts)
    if (sentence && !said.includes(sentence)) said.push(sentence)
  }
  if (pending) {
    // (what was done first, then the question — the last thing said is what STAC waits on)
    const text = [...said, question].join(' ').trim()
    return { text: texts.tail ? `${text} ${texts.tail}`.trim() : text, outcomes, pending }
  }
  let text = said.join(' ')
  const allOk = outcomes.every(o => o.ok)
  const informs = actions.some(a => a.kind !== 'action')
  if (actedOnly && allOk && !informs && texts.done_prefix && !text.toLowerCase().includes(texts.done_prefix.replace(/[.,]/g, '').split(' ').pop()!.toLowerCase()))
    text = `${texts.done_prefix} ${text}`
  if (texts.tail) text = `${text} ${texts.tail}`.trim()
  return { text, outcomes }
}

/** Find one item by what the user called it: the exact name first, then names that hold it. */
export function findByName<T>(items: T[], spoken: string, names: (item: T) => string[]): { item?: T; ambiguous?: T[] } {
  const fold = (s: string) => s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
  const want = fold(spoken)
  if (!want) return {}
  const squash = (s: string) => fold(s).replace(/ /g, '')
  const exact = items.filter(it => names(it).some(n => fold(n) === want || squash(n) === squash(spoken)))
  if (exact.length === 1) return { item: exact[0] }
  if (exact.length > 1) return { ambiguous: exact }
  const partial = items.filter(it => names(it).some(n => { const f = fold(n); return f.includes(want) || want.includes(f) || squash(n).includes(squash(spoken)) }))
  if (partial.length === 1) return { item: partial[0] }
  if (partial.length > 1) return { ambiguous: partial }
  return {}
}
