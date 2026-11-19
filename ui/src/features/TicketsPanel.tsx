/**
 * STAC Build — Tickets: the section of the left panel that lists them, and the modal that marks one.
 *
 * USER 2026-11-09: "una herramienta simple para marcar tickets en cualquier parte del proyecto, como un
 * punto de revisión o una no conformidad". HOW HE WANTS IT (2026-11-19, after trying the first form — a
 * window in Tools: "es muy incomodo"): an object is SELECTED first; a RIGHT CLICK on it gives a menu
 * whose option "ticket" opens a MODAL where it is written ("primero el objeto debe estar seleccionado,
 * luego recien ahi permitir el click derecho, para que no seleccione cualquier cosa por error"). The
 * tickets live in their own section "entre instancias y scans": the whole list with each one's state
 * and history; selecting one takes there; they are shown all or one by one; the number on the model
 * says the rest when the cursor rests on it.
 *
 * The data and its rules are the server's (server/tickets.py): a ticket is a person's statement
 * (`human_stated`), numbered per project, with the geometry epoch of the scan it was marked on.
 */
import { useMemo, useState } from 'react'
import { Check, Eye, EyeOff, LocateFixed, RotateCcw, Trash2 } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { EmptyState } from '../components/ui/EmptyState'
import { Field, Input, SegmentedControl } from '../components/ui/Field'
import { Panel, Row, Stack } from '../components/ui/Panel'
import { useFmt, useT } from '../i18n'

export type TicketKind = 'review' | 'non_conformity'

/** one change of a ticket: who, when, what (the server appends them; a ticket marked before the
 *  history was kept has none and is read from its created / resolved fields) */
export interface TicketEvent { at: string; by: string; action: 'created' | 'resolved' | 'reopened' | 'edited' | 'deleted' }

export interface Ticket {
  id: string
  number: number
  kind: TicketKind
  /** 'deleted': the ticket stays in the list as history (USER 2026-11-19: "si se elimino deberia quedar como
   *  historico no?"), with no pin on the model and nothing more to do with it */
  status: 'open' | 'resolved' | 'deleted'
  title: string
  description: string
  position: number[]
  scan: string | null
  geometry_epoch: number | null
  object: { instance_id: number; label: string } | null
  created_by: string
  created_at: string
  resolved_by: string | null
  resolved_at: string | null
  deleted_by?: string | null
  deleted_at?: string | null
  history?: TicketEvent[]
  stale: boolean
}

export interface TicketCounts { open: number; resolved: number; deleted?: number; review: number; non_conformity: number }

/** a ticket being written: what it says (its point and its object are the right click's) */
export interface TicketDraft { kind: TicketKind; title: string; description: string }

/** how a ticket is drawn: by what it is, unless it is resolved or of a geometry that moved */
export function ticketTone(t: Ticket): 'review' | 'issue' | 'resolved' | 'stale' {
  return t.stale || t.status === 'deleted' ? 'stale' : t.status === 'resolved' ? 'resolved' : t.kind === 'non_conformity' ? 'issue' : 'review'
}

/** the object a ticket is about, as it is named (a segment's label already carries its id) */
export function objectName(o: { instance_id: number; label: string }): string {
  return o.label.includes(`#${o.instance_id}`) ? o.label : `${o.label} #${o.instance_id}`
}

/** a ticket's history, oldest first (from its events, else from what it records of itself) */
export function ticketHistory(t: Ticket): TicketEvent[] {
  if (t.history?.length) return t.history
  const out: TicketEvent[] = [{ at: t.created_at, by: t.created_by, action: 'created' }]
  if (t.resolved_at && t.resolved_by) out.push({ at: t.resolved_at, by: t.resolved_by, action: 'resolved' })
  if (t.deleted_at && t.deleted_by) out.push({ at: t.deleted_at, by: t.deleted_by, action: 'deleted' })
  return out
}

// ── the modal that marks one ────────────────────────────────────────────────────────────────────
interface CreateProps {
  open: boolean
  /** the object the ticket is about (its label and id), for the modal's subtitle */
  objectLabel: string | null
  busy: boolean
  error: string | null
  onCreate: (draft: TicketDraft) => void
  onClose: () => void
}

export function TicketCreateDialog({ open, objectLabel, busy, error, onCreate, onClose }: CreateProps) {
  const t = useT()
  const [kind, setKind] = useState<TicketKind>('review')
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const save = () => { const v = title.trim(); if (v && !busy) onCreate({ kind, title: v, description: description.trim() }) }
  // (each ticket starts empty: the modal is mounted again by its key at every right click)
  return (
    <Dialog open={open} title={t('tickets.newTitle')} subtitle={objectLabel ?? undefined} size="md" busy={busy} onClose={onClose}
      footer={<><Button onClick={onClose} disabled={busy}>{t('common.cancel')}</Button>
        <Button variant="primary" loading={busy} disabled={!title.trim()} onClick={save}>{t('common.save')}</Button></>}>
      <Stack gap={3}>
        {error && <Banner tone="err" compact>{error}</Banner>}
        <Field label={t('tickets.kind')}>
          <SegmentedControl<TicketKind> value={kind} onChange={setKind} ariaLabel={t('tickets.kind')}
            options={[{ value: 'review', label: t('tickets.kind_review') }, { value: 'non_conformity', label: t('tickets.kind_non_conformity') }]} />
        </Field>
        <Field label={t('tickets.titleField')}>
          <Input value={title} maxLength={200} onChange={e => setTitle(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') save() }}
            placeholder={t('tickets.titlePlaceholder')} aria-label={t('tickets.titleField')} data-autofocus />
        </Field>
        <Field label={t('tickets.description')}>
          <Input value={description} maxLength={4000} onChange={e => setDescription(e.target.value)} aria-label={t('tickets.description')} />
        </Field>
      </Stack>
    </Dialog>
  )
}

// ── the section of the left panel ───────────────────────────────────────────────────────────────
interface PanelProps {
  sessionId: string | null
  /** the scan on screen: a ticket of another scan is listed, its pin is not here to go to */
  scanKey: string | null
  tickets: Ticket[]
  counts: TicketCounts | null
  userName: string | null
  /** a manager or an administrator removes any ticket; its author, his own */
  canManage: boolean
  selectedId: string | null
  /** the tickets whose pin is NOT drawn on the model */
  hidden: ReadonlySet<string>
  loadError: string | null
  onSelect: (ticket: Ticket) => void
  onSetHidden: (ids: string[], hidden: boolean) => void
  onSetStatus: (ticket: Ticket, status: 'open' | 'resolved') => void
  onDelete: (ticket: Ticket) => void
}

export function TicketsPanel(p: PanelProps) {
  const t = useT()
  const fmt = useFmt()
  const [showResolved, setShowResolved] = useState(true)
  const [showDeleted, setShowDeleted] = useState(true)
  // open first, then resolved, the deleted last; the newest first in each
  const rank = (x: Ticket) => (x.status === 'open' ? 0 : x.status === 'resolved' ? 1 : 2)
  const rows = useMemo(() => p.tickets
    .filter(x => (x.status === 'open') || (x.status === 'resolved' && showResolved) || (x.status === 'deleted' && showDeleted))
    .sort((a, b) => rank(a) - rank(b) || b.number - a.number), [p.tickets, showResolved, showDeleted])
  const onModel = rows.filter(x => x.status !== 'deleted')           // (a deleted ticket has no pin)
  const allHidden = onModel.length > 0 && onModel.every(x => p.hidden.has(x.id))

  return (
    <Panel title={t('tickets.title')} subtitle={p.counts ? t('tickets.listTitle', { open: p.counts.open, resolved: p.counts.resolved }) : undefined}
      toolbar={p.tickets.length > 0 ? (
        <Row wrap>
          <Button size="sm" icon={allHidden ? <Eye aria-hidden /> : <EyeOff aria-hidden />} disabled={onModel.length === 0} onClick={() => p.onSetHidden(onModel.map(x => x.id), !allHidden)}>
            {allHidden ? t('tickets.showAll') : t('tickets.hideAll')}
          </Button>
          <Button size="sm" active={showResolved} onClick={() => setShowResolved(v => !v)}>{t('tickets.showResolved')}</Button>
          <Button size="sm" active={showDeleted} onClick={() => setShowDeleted(v => !v)}>{t('tickets.showDeleted')}</Button>
        </Row>
      ) : undefined}
      isEmpty={p.tickets.length === 0 && !p.loadError}
      empty={<EmptyState title={t('tickets.emptyTitle')} description={t('tickets.emptyDesc')} />}>
      {p.loadError && <Banner tone="err" compact>{t('tickets.loadFailed', { detail: p.loadError })}</Banner>}
      <ul className="stac-ticketlist">
        {rows.map(r => {
          const here = !r.scan || !p.scanKey || r.scan === p.scanKey
          const gone = r.status === 'deleted'
          const isHidden = gone || p.hidden.has(r.id)
          return (
            <li key={r.id} className={`stac-ticketlist__item ${p.selectedId === r.id ? 'is-selected' : ''} ${isHidden ? 'is-muted' : ''}`.trim()}>
              <button type="button" className="stac-ticketlist__main" onClick={() => p.onSelect(r)} title={here ? t('tickets.focus') : r.scan ?? undefined}>
                <span className={`stac-ticketlist__num stac-ticketlist__num--${ticketTone(r)}`}>{r.number}</span>
                <span className="stac-ticketlist__text">
                  <span className="stac-tickets__title">{r.title}</span>
                  {r.description && <span className="stac-ticketlist__desc">{r.description}</span>}
                </span>
              </button>
              <Row wrap>
                <Badge size="sm" tone={r.kind === 'non_conformity' ? 'err' : 'info'}>{t(`tickets.kind_${r.kind}`)}</Badge>
                <Badge size="sm" tone={r.status === 'resolved' ? 'ok' : gone ? 'neutral' : 'warn'}>{t(`tickets.status_${r.status}`)}</Badge>
                {r.object && <Badge size="sm" tone="neutral">{objectName(r.object)}</Badge>}
                {r.stale && <Badge size="sm" tone="neutral" title={t('tickets.staleHint')}>{t('tickets.stale')}</Badge>}
                {!here && <Badge size="sm" tone="neutral" title={r.scan ?? undefined}>{t('tickets.otherScan')}</Badge>}
              </Row>
              <ul className="stac-ticketlist__history">
                {ticketHistory(r).map((h, i) => (
                  <li key={i}>{t(`tickets.history_${h.action}`, { name: h.by, date: fmt.date(h.at) })}</li>
                ))}
              </ul>
              {!gone && <Row justify="end">
                <IconButton size="sm" label={isHidden ? t('tickets.show') : t('tickets.hide')} icon={isHidden ? <EyeOff aria-hidden /> : <Eye aria-hidden />}
                  onClick={() => p.onSetHidden([r.id], !isHidden)} />
                <IconButton size="sm" label={t('tickets.focus')} icon={<LocateFixed aria-hidden />} disabled={!here} onClick={() => p.onSelect(r)} />
                {p.userName && (r.status === 'open'
                  ? <IconButton size="sm" label={t('tickets.resolve')} icon={<Check aria-hidden />} onClick={() => p.onSetStatus(r, 'resolved')} />
                  : <IconButton size="sm" label={t('tickets.reopen')} icon={<RotateCcw aria-hidden />} onClick={() => p.onSetStatus(r, 'open')} />)}
                {p.userName && (p.canManage || r.created_by === p.userName) &&
                  <IconButton size="sm" variant="danger" label={t('tickets.delete')} icon={<Trash2 aria-hidden />} onClick={() => p.onDelete(r)} />}
              </Row>}
            </li>
          )
        })}
      </ul>
    </Panel>
  )
}
