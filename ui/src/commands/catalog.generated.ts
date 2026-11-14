// GENERATED from server/commands/catalog/*.yaml by `python -m commands.export_ui` — do not edit.
// The commands of the interface STAC can run: their ids and the type of each one's arguments.
// A new command is declared in the catalogue; this file is then regenerated and the compiler
// asks for its executor (ui/src/commands/registry.ts).
export const CATALOG_SHA256 = '84b5dfef45c040b4f6ee372ceccb3a7fbd59d3b1418b2f429e751d00fc8ae6c2'

export const UI_COMMAND_IDS = [
  'talk.set_user_name',
  'talk.forget_user_name',
  'viewer.set_tool',
  'viewer.clear_measurements',
  'viewer.remove_section_box',
  'viewer.show_unsegmented',
  'display.shadows',
  'display.axes',
  'display.grid',
  'display.camera_poses',
  'display.points_as_cubes',
  'display.point_size',
  'display.point_opacity',
  'display.light',
  'display.scene_look',
  'meshes.show',
  'meshes.wireframe',
  'meshes.open_genrecon',
  'app.connect',
  'app.disconnect',
  'app.fullscreen',
  'app.open_settings',
  'app.open_command_palette',
  'app.set_language',
  'app.set_density',
  'layout.show_panel',
  'layout.hide_side_panel',
  'layout.show_inspector',
  'layout.hide_inspector',
  'layout.show_dock',
  'layout.hide_dock',
  'sessions.open',
  'sessions.close',
  'sessions.list',
  'scans.open',
  'scans.list',
  'scans.open_fuse',
  'calibration.measure_distance',
  'calibration.measure_plane',
  'calibration.open',
  'calibration.open_reference',
  'objects.open_library',
  'physics.panel',
  'segmentation.open_autosegment',
  'pipeline.open',
  'flythrough.open',
] as const

export type UiCommandId = typeof UI_COMMAND_IDS[number]

export interface UiCommandArgs {
  'talk.set_user_name': { name: string }
  'talk.forget_user_name': Record<string, never>
  'viewer.set_tool': { tool: 'navigate' | 'measure_distance' | 'measure_angle' | 'section_box' | 'brush' | 'align' }
  'viewer.clear_measurements': Record<string, never>
  'viewer.remove_section_box': Record<string, never>
  'viewer.show_unsegmented': { on: boolean }
  'display.shadows': { on: boolean }
  'display.axes': { on: boolean }
  'display.grid': { on: boolean }
  'display.camera_poses': { on: boolean }
  'display.points_as_cubes': { on: boolean }
  'display.point_size': { size?: number; change?: 'larger' | 'smaller' }
  'display.point_opacity': { opacity: number }
  'display.light': { intensity: number }
  'display.scene_look': Record<string, never>
  'meshes.show': { on: boolean; mesh?: string }
  'meshes.wireframe': { on: boolean }
  'meshes.open_genrecon': Record<string, never>
  'app.connect': Record<string, never>
  'app.disconnect': Record<string, never>
  'app.fullscreen': { on?: boolean }
  'app.open_settings': Record<string, never>
  'app.open_command_palette': Record<string, never>
  'app.set_language': { lang: 'es' | 'en' | 'fr' }
  'app.set_density': { density: 'compact' | 'comfortable' }
  'layout.show_panel': { panel: 'sessions' | 'instances' | 'scans' | 'bim' | 'team' }
  'layout.hide_side_panel': Record<string, never>
  'layout.show_inspector': { tab: 'properties' | 'acta' | 'assistant' }
  'layout.hide_inspector': Record<string, never>
  'layout.show_dock': { tab: 'console' | 'jobs' | 'timeline' | 'report' | 'evaluation' }
  'layout.hide_dock': Record<string, never>
  'sessions.open': { session: string }
  'sessions.close': Record<string, never>
  'sessions.list': Record<string, never>
  'scans.open': { scan: string }
  'scans.list': Record<string, never>
  'scans.open_fuse': Record<string, never>
  'calibration.measure_distance': Record<string, never>
  'calibration.measure_plane': Record<string, never>
  'calibration.open': Record<string, never>
  'calibration.open_reference': { mode: 'distance' | 'plane' }
  'objects.open_library': Record<string, never>
  'physics.panel': { on: boolean }
  'segmentation.open_autosegment': Record<string, never>
  'pipeline.open': { session?: string }
  'flythrough.open': Record<string, never>
}
