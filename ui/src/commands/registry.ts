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
import type { UiCommandArgs, UiCommandId } from './catalog.generated'

/** why a command did not run — worded by the server in the conversation's language (answers.ui_reasons) */
export type Reason = 'no_session' | 'not_connected' | 'server_down' | 'not_found' | 'ambiguous' | 'nothing_to_do' | 'needs_gesture' | 'unavailable' | 'not_wired' | 'failed'

export type Outcome =
  /** `detail`: what the command reports (the session it opened, a list); `state`: for an on / off
   *  command, what it left (false = the "off" wording) */
  | { ok: true; detail?: string; state?: boolean }
  | { ok: false; reason: Reason; detail?: string }

export type Executor<K extends UiCommandId = UiCommandId> = (args: UiCommandArgs[K]) => Outcome | Promise<Outcome>
export type Executors<K extends UiCommandId = UiCommandId> = { [I in K]: Executor<I> }

/** one command of an answer, as the server sends it */
export interface UiAction {
  id: string
  args: Record<string, unknown>
  kind: 'action' | 'query' | 'talk'
  confirm: boolean
  title: string
  done: string
  done_off: string
}

/** the texts an answer carries for the interface to word its commands with */
export interface AnswerTexts {
  done_prefix: string
  name_first?: boolean
  reasons: Record<string, string>
}

const registry = new Map<string, (args: never) => Outcome | Promise<Outcome>>()

/** Register executors (a component registers the commands it owns). Returns the un-register. */
export function registerCommands<K extends UiCommandId>(executors: Executors<K>): () => void {
  const ids = Object.keys(executors) as K[]
  for (const id of ids) registry.set(id, executors[id] as (args: never) => Outcome | Promise<Outcome>)
  return () => { for (const id of ids) if (registry.get(id) === (executors[id] as unknown)) registry.delete(id) }
}

export function isRegistered(id: string): boolean { return registry.has(id) }

/** Run one command. Never throws: a failure is an outcome. */
export async function runCommand(id: string, args: Record<string, unknown>): Promise<Outcome> {
  const ex = registry.get(id)
  if (!ex) return { ok: false, reason: 'not_wired' }
  try {
    return await ex((args ?? {}) as never)
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
export async function runActions(actions: UiAction[], serverAnswer: string, texts: AnswerTexts, actedOnly: boolean): Promise<{ text: string; outcomes: Outcome[] }> {
  const outcomes: Outcome[] = []
  const said: string[] = []
  if (serverAnswer.trim()) said.push(serverAnswer.trim())
  for (const action of actions) {
    const outcome = await runCommand(action.id, action.args)
    outcomes.push(outcome)
    const sentence = wordOutcome(action, outcome, texts)
    if (sentence && !said.includes(sentence)) said.push(sentence)
  }
  let text = said.join(' ')
  const allOk = outcomes.every(o => o.ok)
  const informs = actions.some(a => a.kind !== 'action')
  if (actedOnly && allOk && !informs && texts.done_prefix && !text.toLowerCase().includes(texts.done_prefix.replace(/[.,]/g, '').split(' ').pop()!.toLowerCase()))
    text = `${texts.done_prefix} ${text}`
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
