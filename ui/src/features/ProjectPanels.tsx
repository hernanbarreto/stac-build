/**
 * STAC Build — the project's DOCUMENTATION and its GANTT: two sections of the left panel.
 *
 * USER 2026-11-19: "al panel izquiero le vamos a agregar dos campos mas, uno es de docuemntacion, donde ahi
 * vamos a tener documentacion de contrato, ingenieria, matriz de requisitos, etc, es un arbol completo, pero
 * por el momento vacio, y otro campo mas que es el gantt". The tree's branches are his; nothing is in them
 * yet and nothing here reads or writes the project: both are the PLACE these will live in.
 */
import { ChartNoAxesGantt, FileText, FolderClosed } from 'lucide-react'
import { EmptyState } from '../components/ui/EmptyState'
import { Panel } from '../components/ui/Panel'
import { Tree, type TreeNodeData } from '../components/ui/Tree'
import { useT } from '../i18n'

/** the branches of the documentation's tree (each one empty for now) */
const BRANCHES = ['contract', 'engineering', 'requirements'] as const

export function DocumentationPanel() {
  const t = useT()
  const nodes: TreeNodeData<string>[] = BRANCHES.map(b => ({
    id: b, data: b, icon: <FolderClosed aria-hidden />, label: t(`docs.branch_${b}`), meta: t('docs.empty'), defaultExpanded: false, children: [],
  }))
  return (
    <Panel title={t('docs.title')} subtitle={t('docs.subtitle')}>
      <Tree ariaLabel={t('docs.title')} dense nodes={nodes} selectedId={null} onSelect={() => undefined} />
      <EmptyState compact icon={<FileText aria-hidden />} title={t('docs.emptyTitle')} description={t('docs.emptyDesc')} />
    </Panel>
  )
}

export function GanttPanel() {
  const t = useT()
  return (
    <Panel title={t('gantt.title')} isEmpty
      empty={<EmptyState icon={<ChartNoAxesGantt aria-hidden />} title={t('gantt.emptyTitle')} description={t('gantt.emptyDesc')} />} />
  )
}
