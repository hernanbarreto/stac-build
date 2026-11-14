// GENERATED from server/commands/catalog/*.yaml by `python -m commands.export_ui` — do not edit.
// The commands of the interface STAC can run: their ids and the type of each one's arguments.
// A new command is declared in the catalogue; this file is then regenerated and the compiler
// asks for its executor (ui/src/commands/registry.ts).
export const CATALOG_SHA256 = 'e5c5e503f2c7f166dc2f344a6320fdccde5a16d565f946bd6993640b1fda971d'

export const UI_COMMAND_IDS = [
  'talk.set_user_name',
  'talk.forget_user_name',
  'talk.spoken_answers',
  'talk.hands_free',
  'volumes.focus',
  'volumes.set_visible',
  'volumes.set_obb',
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
  'display.set_precision_layer',
  'display.set_background',
  'display.set_grid_style',
  'display.reset_scene_look',
  'display.set_point_budget',
  'display.set_confidence_threshold',
  'display.set_votes_threshold',
  'display.hide_below_votes',
  'meshes.show',
  'meshes.wireframe',
  'meshes.open_genrecon',
  'meshes.focus',
  'app.connect',
  'app.disconnect',
  'app.fullscreen',
  'app.open_settings',
  'app.open_command_palette',
  'app.set_language',
  'app.set_density',
  'app.logout',
  'layout.show_panel',
  'layout.hide_side_panel',
  'layout.show_inspector',
  'layout.hide_inspector',
  'layout.show_dock',
  'layout.hide_dock',
  'sessions.open',
  'sessions.close',
  'sessions.list',
  'sessions.select',
  'scans.open',
  'scans.list',
  'scans.open_fuse',
  'scans.close_tab',
  'scans.cycle_tab',
  'calibration.measure_distance',
  'calibration.measure_plane',
  'calibration.open',
  'calibration.open_reference',
  'calibration.close',
  'objects.open_library',
  'objects.focus',
  'objects.set_gizmo_mode',
  'objects.align',
  'objects.set_visible',
  'objects.set_obb',
  'physics.panel',
  'physics.set_ball',
  'physics.toggle_placing',
  'physics.drop',
  'physics.clear',
  'segmentation.open_autosegment',
  'pipeline.open',
  'flythrough.open',
  'flythrough.close',
  'brush.set_shape',
  'brush.set_size',
  'brush.set_rotation',
  'brush.preview_confidence',
  'brush.cancel_confidence_preview',
  'brush.clear_marks',
  'brush.undo',
  'instances.cubes_set_visible',
  'instances.cubes_set_all_visible',
  'instances.cubes_focus',
  'admin.open',
] as const

export type UiCommandId = typeof UI_COMMAND_IDS[number]

export interface UiCommandArgs {
  'talk.set_user_name': { name: string }
  'talk.forget_user_name': Record<string, never>
  'talk.spoken_answers': { on: boolean }
  'talk.hands_free': { on: boolean }
  'volumes.focus': { volume_id: number }
  'volumes.set_visible': { volume_id: number; on: boolean }
  'volumes.set_obb': { volume_id: number; on: boolean }
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
  'display.set_precision_layer': { layer: 'flyers' | 'mono_detail' | 'mixed_unresolved'; on: boolean }
  'display.set_background': { kind: 'default' | 'horizon' | 'plain'; color?: string }
  'display.set_grid_style': { cell_m?: 0.1 | 0.25 | 0.5 | 1 | 2 | 5; major_every?: 2 | 4 | 5 | 10; color?: string }
  'display.reset_scene_look': Record<string, never>
  'display.set_point_budget': { millions: number }
  'display.set_confidence_threshold': { threshold: number }
  'display.set_votes_threshold': { views: number }
  'display.hide_below_votes': { on: boolean }
  'meshes.show': { on: boolean; mesh?: string }
  'meshes.wireframe': { on: boolean }
  'meshes.open_genrecon': Record<string, never>
  'meshes.focus': { mesh: string }
  'app.connect': Record<string, never>
  'app.disconnect': Record<string, never>
  'app.fullscreen': { on?: boolean }
  'app.open_settings': Record<string, never>
  'app.open_command_palette': Record<string, never>
  'app.set_language': { lang: 'es' | 'en' | 'fr' }
  'app.set_density': { density: 'compact' | 'comfortable' }
  'app.logout': Record<string, never>
  'layout.show_panel': { panel: 'sessions' | 'instances' | 'scans' | 'bim' | 'team' }
  'layout.hide_side_panel': Record<string, never>
  'layout.show_inspector': { tab: 'properties' | 'acta' | 'assistant' }
  'layout.hide_inspector': Record<string, never>
  'layout.show_dock': { tab: 'console' | 'jobs' | 'timeline' | 'report' | 'evaluation' }
  'layout.hide_dock': Record<string, never>
  'sessions.open': { session: string }
  'sessions.close': Record<string, never>
  'sessions.list': Record<string, never>
  'sessions.select': { session: string }
  'scans.open': { scan: string }
  'scans.list': Record<string, never>
  'scans.open_fuse': Record<string, never>
  'scans.close_tab': { scan?: string }
  'scans.cycle_tab': { direction: 'next' | 'prev' }
  'calibration.measure_distance': Record<string, never>
  'calibration.measure_plane': Record<string, never>
  'calibration.open': Record<string, never>
  'calibration.open_reference': { mode: 'distance' | 'plane' }
  'calibration.close': Record<string, never>
  'objects.open_library': Record<string, never>
  'objects.focus': { id: number | string }
  'objects.set_gizmo_mode': { mode: 'translate' | 'rotate' | 'scale' }
  'objects.align': { op: 'floor' | 'same_base' | 'on_top' | 'center_xz' | 'center_y'; target?: string }
  'objects.set_visible': { id: number | string; on: boolean }
  'objects.set_obb': { id: number | string; on: boolean }
  'physics.panel': { on: boolean }
  'physics.set_ball': { diameter_cm?: number; bounce?: number }
  'physics.toggle_placing': { on: boolean }
  'physics.drop': Record<string, never>
  'physics.clear': Record<string, never>
  'segmentation.open_autosegment': Record<string, never>
  'pipeline.open': { session?: string }
  'flythrough.open': Record<string, never>
  'flythrough.close': Record<string, never>
  'brush.set_shape': { shape: 'sphere' | 'cube' | 'box' }
  'brush.set_size': { radius_m: number }
  'brush.set_rotation': { yaw_deg: number }
  'brush.preview_confidence': { threshold: number }
  'brush.cancel_confidence_preview': Record<string, never>
  'brush.clear_marks': Record<string, never>
  'brush.undo': Record<string, never>
  'instances.cubes_set_visible': { id: number; on: boolean }
  'instances.cubes_set_all_visible': { on: boolean }
  'instances.cubes_focus': { id: number }
  'admin.open': Record<string, never>
}
