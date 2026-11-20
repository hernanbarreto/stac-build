/**
 * StatusBar — 28 px readout strip (§5, §7): connection, session, last
 * message, units, geometric epoch, cursor X Y Z (mono, 3 decimals, m) ,
 * job, points / fps. Every number carries its unit.
 */
import { Terminal } from 'lucide-react'
import { useFmt, useT } from '../i18n'
import { Sep } from '../components/ui/Panel'
import { Badge } from '../components/ui/Badge'

export interface StatusBarProps {
  serverAlive: boolean
  connected: boolean
  sessionLabel: string | null
  scanLabel?: string | null
  message: string
  messageTone?: 'neutral' | 'ok' | 'warn' | 'err'
  epoch: number | null
  storedEpochs?: number
  cursor: { x: number; y: number; z: number } | null
  job: string | null
  jobPct?: number | null
  points: number
  fps: number
  onToggleConsole: () => void
  consoleOpen: boolean
}

export function StatusBar({ serverAlive, connected, sessionLabel, scanLabel, message, messageTone = 'neutral', epoch, storedEpochs = 0, cursor, job, jobPct, points, fps, onToggleConsole, consoleOpen }: StatusBarProps) {
  const t = useT()
  const fmt = useFmt()
  const coord = (v: number) => fmt.number(v, 3)
  return (
    <footer className="stac-status" role="contentinfo">
      <span className="stac-status__item" title={serverAlive ? t('status.serverUp') : t('status.serverDown')}>
        <span className={`stac-live-dot ${serverAlive ? 'stac-live-dot--ok stac-live-dot--pulse' : 'stac-live-dot--err'}`} aria-hidden />
        <span>{connected ? t('status.connected') : serverAlive ? t('status.disconnected') : t('status.serverDown')}</span>
      </span>
      <Sep />
      <span className="stac-status__item stac-status__item--grow" title={message}>
        <span className="stac-status__session">{sessionLabel ?? t('status.noSession')}</span>
        {scanLabel && <span className="stac-status__group stac-status__group--scan"><Sep /><span className="stac-status__scan">{scanLabel}</span></span>}
        {message && <><Sep /><span className={`stac-status__message stac-status__message--${messageTone}`}>{message}</span></>}
      </span>
      {job && (
        <span className="stac-status__item" title={job}>
          <Badge tone="brand" size="sm" dot>{job}{jobPct != null ? ` ${fmt.percent(jobPct / 100)}` : ''}</Badge>
        </span>
      )}
      {/* each readout with its separator, in a group the page's width can drop whole (regions.css): on a narrow
          page the bar keeps the connection, the session, the running job and the console's button */}
      <span className="stac-status__group stac-status__group--units">
        <Sep />
        <span className="stac-status__item stac-mono" title={t('status.units')}>{t('status.metres')}</span>
      </span>
      <span className="stac-status__group stac-status__group--epoch">
        <Sep />
        <span className="stac-status__item stac-mono" title={t('status.epochTitle')}>
          {t('status.epoch')} {epoch != null ? fmt.integer(epoch) : t('status.dash')}
          {storedEpochs > 0 && <Badge tone="info" size="sm">{t('status.storedEpochs', { count: storedEpochs })}</Badge>}
        </span>
      </span>
      <span className="stac-status__group stac-status__group--cursor">
        <Sep />
        <span className="stac-status__item stac-status__cursor stac-mono" title={t('status.cursorTitle')} aria-live="off">
          {cursor ? `X ${coord(cursor.x)}  Y ${coord(cursor.y)}  Z ${coord(cursor.z)} m` : `X ${t('status.dash')}  Y ${t('status.dash')}  Z ${t('status.dash')}`}
        </span>
      </span>
      <span className="stac-status__group stac-status__group--points">
        <Sep />
        <span className="stac-status__item stac-mono" title={t('status.pointsTitle')}>
          {points > 0 ? t('status.points', { n: fmt.millions(points) }) : t('status.noPoints')}
          {fps > 0 && <> <Sep /> {t('status.fps', { n: fmt.integer(fps) })}</>}
        </span>
      </span>
      <Sep />
      <button type="button" className={`stac-status__btn ${consoleOpen ? 'stac-status__btn--on' : ''}`.trim()} onClick={onToggleConsole} aria-pressed={consoleOpen} title={t('dock.console')}>
        <Terminal aria-hidden /><span>{t('dock.console')}</span>
      </button>
    </footer>
  )
}
