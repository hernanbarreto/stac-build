// GENERATED from server/commands/catalog/*.yaml by `python -m commands.export_ui` — do not edit.
// The commands of the interface STAC can run: their ids and the type of each one's arguments.
// A new command is declared in the catalogue; this file is then regenerated and the compiler
// asks for its executor (ui/src/commands/registry.ts).
export const CATALOG_SHA256 = '0eb25577c71ea07016806f9c7c7ed9a035bb99b35a1fecb8b4b16465555f0b10'

export const UI_COMMAND_IDS = [
  'talk.set_user_name',
  'talk.forget_user_name',
  'talk.spoken_answers',
  'talk.hands_free',
  'measure.finish_plane',
  'measure.cancel_pending',
  'volumes.focus',
  'volumes.set_visible',
  'volumes.set_obb',
  'volumes.delete',
  'volumes.set_gizmo_mode',
  'volumes.set_solid',
  'viewer.set_tool',
  'viewer.clear_measurements',
  'viewer.remove_section_box',
  'viewer.show_unsegmented',
  'viewer.orient_view',
  'viewer.fly_to_camera_pose',
  'viewer.fly_to_point',
  'viewer.frame_box',
  'viewer.align_set_gizmo_mode',
  'viewer.align_save',
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
  'meshes.delete',
  'meshes.focus',
  'meshes.run_genrecon',
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
  'layout.resize_panel',
  'sessions.open',
  'sessions.close',
  'sessions.list',
  'sessions.select',
  'sessions.create',
  'sessions.rename',
  'sessions.delete',
  'sessions.filter',
  'scans.open',
  'scans.list',
  'scans.open_fuse',
  'scans.close_tab',
  'scans.cycle_tab',
  'scans.set_reference',
  'calibration.measure_distance',
  'calibration.measure_plane',
  'calibration.open',
  'calibration.open_reference',
  'calibration.calibrate_readout',
  'calibration.save_distance',
  'calibration.save_plane',
  'calibration.toggle_role',
  'calibration.delete_measurement',
  'calibration.delete_plane',
  'calibration.set_mode',
  'calibration.close',
  'calibration.apply',
  'objects.open_library',
  'objects.add',
  'objects.focus',
  'objects.set_gizmo_mode',
  'objects.align',
  'objects.remove_selected',
  'objects.remove',
  'objects.rename',
  'objects.set_visible',
  'objects.set_obb',
  'physics.panel',
  'physics.set_ball',
  'physics.toggle_placing',
  'physics.drop',
  'physics.clear',
  'segmentation.open_autosegment',
  'pipeline.open',
  'pipeline.cancel',
  'flythrough.open',
  'flythrough.play_pause',
  'flythrough.step_frame',
  'flythrough.seek',
  'flythrough.set_speed',
  'flythrough.close',
  'brush.set_shape',
  'brush.set_size',
  'brush.set_rotation',
  'brush.preview_confidence',
  'brush.cancel_confidence_preview',
  'brush.set_box_gizmo_mode',
  'brush.remove_box',
  'brush.clear_marks',
  'brush.undo',
  'instances.rename',
  'instances.delete',
  'instances.filter',
  'instances.set_floor_level',
  'instances.cubes_set_visible',
  'instances.cubes_set_all_visible',
  'instances.cubes_focus',
  'bim.load_ifc',
  'bim.delete_ifc',
  'bim.set_element_visible',
  'bim.select_element',
  'bim.set_element_opacity',
  'bim.filter',
  'bim.ctx_set_transparency',
  'bim.ctx_hide',
  'bim.ctx_isolate',
  'bim.ctx_show_all',
  'deviation.open_compare',
  'deviation.run_comparison',
  'deviation.toggle_view',
  'certify.set_epoch_layers',
  'certify.show_trajectory_edges',
  'certify.select_epoch',
  'certify.set_section_tab',
  'evaluation.refresh_runs',
  'evaluation.select_run',
  'evaluation.open_file',
  'evaluation.set_json_view',
  'admin.open',
  'reports.run_deviation',
] as const

export type UiCommandId = typeof UI_COMMAND_IDS[number]

// the ones STAC runs only after a spoken confirmation: their executor is a { prepare, run } pair
export const UI_CONFIRM_IDS = [
  'volumes.delete',
  'viewer.align_save',
  'meshes.delete',
  'meshes.run_genrecon',
  'sessions.delete',
  'calibration.delete_measurement',
  'calibration.delete_plane',
  'calibration.apply',
  'objects.remove_selected',
  'objects.remove',
  'pipeline.cancel',
  'instances.delete',
  'instances.set_floor_level',
  'bim.delete_ifc',
  'deviation.run_comparison',
  'reports.run_deviation',
] as const

export type UiConfirmId = typeof UI_CONFIRM_IDS[number]

export interface UiCommandArgs {
  'talk.set_user_name': { name: string }
  'talk.forget_user_name': Record<string, never>
  'talk.spoken_answers': { on: boolean }
  'talk.hands_free': { on: boolean }
  'measure.finish_plane': Record<string, never>
  'measure.cancel_pending': Record<string, never>
  'volumes.focus': { volume_id: number }
  'volumes.set_visible': { volume_id: number; on: boolean }
  'volumes.set_obb': { volume_id: number; on: boolean }
  'volumes.delete': { volume_id: number }
  'volumes.set_gizmo_mode': { mode: 'translate' | 'rotate' | 'scale' }
  'volumes.set_solid': { on: boolean }
  'viewer.set_tool': { tool: 'navigate' | 'measure_distance' | 'measure_angle' | 'section_box' | 'brush' | 'align' }
  'viewer.clear_measurements': Record<string, never>
  'viewer.remove_section_box': Record<string, never>
  'viewer.show_unsegmented': { on: boolean }
  'viewer.orient_view': { dir: unknown[] }
  'viewer.fly_to_camera_pose': { keyframe: number }
  'viewer.fly_to_point': { position: unknown[]; radius?: number }
  'viewer.frame_box': { min: unknown[]; max: unknown[] }
  'viewer.align_set_gizmo_mode': { mode: 'rotate' | 'translate' }
  'viewer.align_save': Record<string, never>
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
  'meshes.delete': { mesh: string }
  'meshes.focus': { mesh: string }
  'meshes.run_genrecon': { confidence: number }
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
  'layout.resize_panel': { panel: 'left' | 'inspector' | 'dock'; size_px: number }
  'sessions.open': { session: string }
  'sessions.close': Record<string, never>
  'sessions.list': Record<string, never>
  'sessions.select': { session: string }
  'sessions.create': { name: string }
  'sessions.rename': { session: string; name: string }
  'sessions.delete': { session: string }
  'sessions.filter': { text: string }
  'scans.open': { scan: string }
  'scans.list': Record<string, never>
  'scans.open_fuse': Record<string, never>
  'scans.close_tab': { scan?: string }
  'scans.cycle_tab': { direction: 'next' | 'prev' }
  'scans.set_reference': { scan: string }
  'calibration.measure_distance': Record<string, never>
  'calibration.measure_plane': Record<string, never>
  'calibration.open': Record<string, never>
  'calibration.open_reference': { mode: 'distance' | 'plane' }
  'calibration.calibrate_readout': { readout_id?: string }
  'calibration.save_distance': { value_m: number; sigma_m?: number; role: 'gauge' | 'test'; tool: 'tape' | 'laser' | 'total_station' | 'other_scan'; reference_scan?: string; note?: string }
  'calibration.save_plane': { label?: string; note?: string }
  'calibration.toggle_role': { id: string; role: 'gauge' | 'test' }
  'calibration.delete_measurement': { id: string }
  'calibration.delete_plane': { id: string }
  'calibration.set_mode': { mode: 'global' | 'axes' }
  'calibration.close': Record<string, never>
  'calibration.apply': Record<string, never>
  'objects.open_library': Record<string, never>
  'objects.add': { name: string }
  'objects.focus': { id: number | string }
  'objects.set_gizmo_mode': { mode: 'translate' | 'rotate' | 'scale' }
  'objects.align': { op: 'floor' | 'same_base' | 'on_top' | 'center_xz' | 'center_y'; target?: string }
  'objects.remove_selected': Record<string, never>
  'objects.remove': { id: number | string }
  'objects.rename': { id: number | string; name: string }
  'objects.set_visible': { id: number | string; on: boolean }
  'objects.set_obb': { id: number | string; on: boolean }
  'physics.panel': { on: boolean }
  'physics.set_ball': { diameter_cm?: number; bounce?: number }
  'physics.toggle_placing': { on: boolean }
  'physics.drop': Record<string, never>
  'physics.clear': Record<string, never>
  'segmentation.open_autosegment': Record<string, never>
  'pipeline.open': { session?: string }
  'pipeline.cancel': Record<string, never>
  'flythrough.open': Record<string, never>
  'flythrough.play_pause': { on?: boolean }
  'flythrough.step_frame': { frames: number }
  'flythrough.seek': { seconds: number }
  'flythrough.set_speed': { rate: 0.25 | 0.5 | 1 | 2 }
  'flythrough.close': Record<string, never>
  'brush.set_shape': { shape: 'sphere' | 'cube' | 'box' }
  'brush.set_size': { radius_m: number }
  'brush.set_rotation': { yaw_deg: number }
  'brush.preview_confidence': { threshold: number }
  'brush.cancel_confidence_preview': Record<string, never>
  'brush.set_box_gizmo_mode': { mode: 'translate' | 'rotate' | 'scale' }
  'brush.remove_box': Record<string, never>
  'brush.clear_marks': Record<string, never>
  'brush.undo': Record<string, never>
  'instances.rename': { id: number | string; label: string }
  'instances.delete': { id: number | string }
  'instances.filter': { text: string }
  'instances.set_floor_level': { id: number | string }
  'instances.cubes_set_visible': { id: number; on: boolean }
  'instances.cubes_set_all_visible': { on: boolean }
  'instances.cubes_focus': { id: number }
  'bim.load_ifc': Record<string, never>
  'bim.delete_ifc': { filename: string }
  'bim.set_element_visible': { element: number | string; on: boolean }
  'bim.select_element': { element: number | string }
  'bim.set_element_opacity': { element: number | string; opacity: number }
  'bim.filter': { text: string }
  'bim.ctx_set_transparency': { element: number | string; transparency: number }
  'bim.ctx_hide': { element: number | string }
  'bim.ctx_isolate': { element: number | string }
  'bim.ctx_show_all': Record<string, never>
  'deviation.open_compare': Record<string, never>
  'deviation.run_comparison': { registration: 'auto' | 'manual' }
  'deviation.toggle_view': { on: boolean }
  'certify.set_epoch_layers': { before?: boolean; after?: boolean }
  'certify.show_trajectory_edges': { odometry?: boolean; loops?: boolean }
  'certify.select_epoch': { epoch: number }
  'certify.set_section_tab': { tab: 'attention' | 'edges' | 'duplicates' | 'acta' }
  'evaluation.refresh_runs': Record<string, never>
  'evaluation.select_run': { run_id: string }
  'evaluation.open_file': { path: string }
  'evaluation.set_json_view': { mode: 'structured' | 'raw' }
  'admin.open': Record<string, never>
  'reports.run_deviation': { tolerance_mm: number }
}
