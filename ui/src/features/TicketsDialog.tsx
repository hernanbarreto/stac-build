/**
 * TicketsDialog — the points of the model marked as something to look at
 * (USER 2026-11-09: "TICKETS — una herramienta simple para marcar tickets en cualquier parte del
 * proyecto, como un punto de revisión o una no conformidad: es para mostrar el concepto nomás").
 *
 * A ticket is marked in two steps: here its kind (a review item / a non-conformity) and its title,
 * then ONE CLICK on the model — the point is the user's, never guessed. The list is the PROJECT's
 * (one numbering for all its scans); each ticket shows what it is, who marked it, the object its
 * point lies on, and whether it is open or resolved (by whom). A ticket of another scan of the
 * project is listed and not drawn; one whose scan's geometry moved since it was marked says so.
 *
 * Presentational: the tickets, the placing state and every action come from App (the click on the
 * model arrives there).
 */
import { useMemo, useState } from 'react'
import { Check, Crosshair, LocateFixed, RotateCcw, Trash2, X } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { Field, Input, SegmentedControl } from '../components/ui/Field'
import { Row, Section, Stack } from '../components/ui/Panel'
import { Table, type Column } from '../components/ui/Table'
import { useFmt, useT } from '../i18n'

export type TicketKind = 'review' | 'non_conformity'

export interface Ticket {
  id: string
  number: number
  kind: TicketKind
  status: 'open' | 'resolved'
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
  stale: boolean
}

export interface TicketCounts { open: number; resolved: number; review: number; non_conformity: number }

/** a ticket being marked: what it says, waiting for its point */
export interface TicketDraft { kind: TicketKind; title: string; description: string }

interface Props {
  open: boolean
  sessionId: string | null
  scanKey: string | null
  tickets: Ticket[]
  counts: TicketCounts | null
  /** signed in: a ticket records who marked it */
  userName: string | null
  canManage: boolean
  placing: TicketDraft | null
  selectedId: string | null
  loadError: string | null
  onStartPlacing: (draft: TicketDraft) => void
  onCancelPlacing: () => void
  onSelect: (id: string | null) => void
  onFocus: (ticket: Ticket) => void
  onSetStatus: (ticket: Ticket, status: 'open' | 'resolved') => void
  onDelete: (ticket: Ticket) => void
  onClose: () => void
}

/** how a ticket is drawn: by what it is, unless it is resolved or of a geometry that moved */
export function ticketTone(t: Ticket): 'review' | 'issue' | 'resolved' | 'stale' {
  return t.stale ? 'stale' : t.status === 'resolved' ? 'resolved' : t.kind === 'non_conformity' ? 'issue' : 'review'
}

export function TicketsDialog(p: Props) {
  const t = useT()
  const fmt = useFmt()
  const [kind, setKind] = useState<TicketKind>('review')
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [showResolved, setShowResolved] = useState(true)

  const rows = useMemo(() => p.tickets
    .filter(x => showResolved || x.status === 'open')
    .sort((a, b) => Number(a.status === 'resolved') - Number(b.status === 'resolved') || b.number - a.number), [p.tickets, showResolved])

  const start = () => {
    const v = title.trim()
    if (!v) return
    p.onStartPlacing({ kind, title: v, description: description.trim() })
  }

  const columns: Column<Ticket>[] = useMemo(() => [
    { id: 'number', header: '#', mono: true, width: '3rem', sortValue: r => r.number, cell: r => r.number },
    { id: 'kind', header: t('tickets.kind'), sortValue: r => r.kind,
      cell: r => <Badge size="sm" tone={r.kind === 'non_conformity' ? 'err' : 'info'}>{t(`tickets.kind_${r.kind}`)}</Badge> },
    { id: 'title', header: t('tickets.titleField'), sortValue: r => r.title, cell: r => (
        <Stack gap={1}>
          <span className="stac-tickets__title" title={r.description || r.title}>{r.title}</span>
          <span className="stac-tickets__meta">
            {[r.object ? `${r.object.label}#${r.object.instance_id}` : null, r.created_by, fmt.date(r.created_at)].filter(Boolean).join(', ')}
          </span>
        </Stack>
      ) },
    { id: 'status', header: t('tickets.status'), sortValue: r => r.status, cell: r => (
        <Row>
          <Badge size="sm" tone={r.status === 'resolved' ? 'ok' : 'warn'} title={r.resolved_by ? t('tickets.resolvedBy', { name: r.resolved_by }) : undefined}>{t(`tickets.status_${r.status}`)}</Badge>
          {r.stale && <Badge size="sm" tone="neutral" title={t('tickets.staleHint')}>{t('tickets.stale')}</Badge>}
          {r.scan && p.scanKey && r.scan !== p.scanKey && <Badge size="sm" tone="neutral" title={r.scan}>{t('tickets.otherScan')}</Badge>}
        </Row>
      ) },
    { id: 'actions', header: '', align: 'right', cell: r => (
        <Row justify="end">
          <IconButton size="sm" label={t('tickets.focus')} icon={<LocateFixed aria-hidden />} disabled={!!r.scan && !!p.scanKey && r.scan !== p.scanKey}
            onClick={e => { e.stopPropagation(); p.onFocus(r) }} />
          {p.userName && (r.status === 'open'
            ? <IconButton size="sm" label={t('tickets.resolve')} icon={<Check aria-hidden />} onClick={e => { e.stopPropagation(); p.onSetStatus(r, 'resolved') }} />
            : <IconButton size="sm" label={t('tickets.reopen')} icon={<RotateCcw aria-hidden />} onClick={e => { e.stopPropagation(); p.onSetStatus(r, 'open') }} />)}
          {p.userName && (p.canManage || r.created_by === p.userName) &&
            <IconButton size="sm" variant="danger" label={t('tickets.delete')} icon={<Trash2 aria-hidden />} onClick={e => { e.stopPropagation(); p.onDelete(r) }} />}
        </Row>
      ) },
  ], [t, fmt, p])

  return (
    <Dialog open={p.open} title={t('tickets.title')} subtitle={[p.sessionId, p.scanKey].filter(Boolean).join(', ') || undefined}
      size="lg" nonBlocking position="right" onClose={p.onClose}>
      <Stack gap={3}>
        {p.loadError && <Banner tone="err">{t('tickets.loadFailed', { detail: p.loadError })}</Banner>}
        <Section title={t('tickets.newTitle')}>
          {p.placing ? (
            // the second step: the point, on the model
            <Banner tone="info" icon={<Crosshair aria-hidden />}
              action={<Button size="sm" icon={<X aria-hidden />} onClick={p.onCancelPlacing}>{t('common.cancel')}</Button>}>
              {t('tickets.placing', { title: p.placing.title })}
            </Banner>
          ) : !p.userName ? (
            <Banner tone="warn" compact>{t('tickets.signIn')}</Banner>
          ) : (
            <Stack gap={2}>
              <Field label={t('tickets.kind')}>
                <SegmentedControl<TicketKind> value={kind} onChange={setKind} ariaLabel={t('tickets.kind')}
                  options={[{ value: 'review', label: t('tickets.kind_review') }, { value: 'non_conformity', label: t('tickets.kind_non_conformity') }]} />
              </Field>
              <Field label={t('tickets.titleField')}>
                <Input value={title} maxLength={200} onChange={e => setTitle(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') start() }}
                  placeholder={t('tickets.titlePlaceholder')} aria-label={t('tickets.titleField')} data-autofocus />
              </Field>
              <Field label={t('tickets.description')}>
                <Input value={description} maxLength={4000} onChange={e => setDescription(e.target.value)} aria-label={t('tickets.description')} />
              </Field>
              <Row justify="end">
                <Button variant="primary" icon={<Crosshair aria-hidden />} disabled={!title.trim()} title={t('tickets.markHint')} onClick={start}>{t('tickets.mark')}</Button>
              </Row>
            </Stack>
          )}
        </Section>
        <Section title={p.counts ? t('tickets.listTitle', { open: p.counts.open, resolved: p.counts.resolved }) : t('tickets.title')}
          actions={<Button size="sm" active={showResolved} onClick={() => setShowResolved(v => !v)}>{t('tickets.showResolved')}</Button>}>
          <Table<Ticket> columns={columns} rows={rows} rowKey={r => r.id} selectedKey={p.selectedId ?? undefined}
            onRowClick={r => { p.onSelect(r.id); if (!r.scan || !p.scanKey || r.scan === p.scanKey) p.onFocus(r) }}
            emptyTitle={t('tickets.emptyTitle')} emptyDescription={t('tickets.emptyDesc')} />
        </Section>
      </Stack>
    </Dialog>
  )
}
