
from facefusion import config, state_manager
from facefusion.ffmpeg import get_available_encoder_set
import tempfile
import facefusion.choices
from facefusion.processors.modules.face_swapper import choices as face_swapper_choices
from facefusion.common_helper import get_first, get_last
from facefusion.execution import get_available_execution_providers


def setup_config_path_args(config_path) -> dict:
    state_manager.init_item('config_path', config_path)
    return {
        'config_path': config_path
    }


def setup_temp_path_args() -> dict:
    return {
        'temp_path': config.get_str_value('paths', 'temp_path', tempfile.gettempdir())
    }


def setup_jobs_path_args() -> dict:
    return {
        'jobs_path': config.get_str_value('paths', 'jobs_path', '.jobs')
    }


def setup_source_paths_args() -> dict:
    return {
        'source_paths': config.get_str_list('paths', 'source_paths')
    }


def setup_target_path_args() -> dict:
    return {
        'target_path': config.get_str_value('paths', 'target_path')
    }


def setup_output_path_args() -> dict:
    return {
        'output_path': config.get_str_value('paths', 'output_path')
    }


def setup_face_detector_args() -> dict:
    model = config.get_str_value('face_detector', 'face_detector_model', 'yolo_face')
    size_choices = facefusion.choices.face_detector_set.get(model)
    size = config.get_str_value('face_detector', 'face_detector_size', get_last(size_choices))
    margin = config.get_int_list('face_detector', 'face_detector_margin', '0 0 0 0')
    angles = config.get_int_list('face_detector', 'face_detector_angles', '0')
    score = config.get_float_value('face_detector', 'face_detector_score', '0.5')

    return {
        'face_detector_model': model,
        'face_detector_size': size,
        'face_detector_margin': margin,
        'face_detector_angles': angles,
        'face_detector_score': score
    }


def setup_face_landmarker_args() -> dict:
    return {
        'face_landmarker_model': config.get_str_value('face_landmarker', 'face_landmarker_model', '2dfan4'),
        'face_landmarker_score': config.get_float_value('face_landmarker', 'face_landmarker_score', '0.5')
    }


def setup_face_selector_args() -> dict:
    return {
        'face_selector_mode': config.get_str_value('face_selector', 'face_selector_mode', 'reference'),
        'face_selector_order': config.get_str_value('face_selector', 'face_selector_order', 'large-small'),
        'face_selector_age_start': config.get_int_value('face_selector', 'face_selector_age_start'),
        'face_selector_age_end': config.get_int_value('face_selector', 'face_selector_age_end'),
        'face_selector_gender': config.get_str_value('face_selector', 'face_selector_gender'),
        'face_selector_race': config.get_str_value('face_selector', 'face_selector_race'),
        'reference_face_position': config.get_int_value('face_selector', 'reference_face_position', '0'),
        'reference_face_distance': config.get_float_value('face_selector', 'reference_face_distance', '0.3'),
        'reference_frame_number': config.get_int_value('face_selector', 'reference_frame_number', '0')
    }


def setup_face_masker_args() -> dict:
    return {
        'face_occluder_model': config.get_str_value('face_masker', 'face_occluder_model', 'xseg_1'),
        'face_parser_model': config.get_str_value('face_masker', 'face_parser_model', 'bisenet_resnet_34'),
        'face_mask_types': config.get_str_list('face_masker', 'face_mask_types', 'box'),
        'face_mask_areas': config.get_str_list('face_masker', 'face_mask_areas', ' '.join(facefusion.choices.face_mask_areas)),
        'face_mask_regions': config.get_str_list('face_masker', 'face_mask_regions', ' '.join(facefusion.choices.face_mask_regions)),
        'face_mask_blur': config.get_float_value('face_masker', 'face_mask_blur', '0.3'),
        'face_mask_padding': config.get_int_list('face_masker', 'face_mask_padding', '0 0 0 0')
    }


def setup_voice_extractor_args() -> dict:
    return {
        'voice_extractor_model': config.get_str_value('voice_extractor', 'voice_extractor_model', 'kim_vocal_2')
    }


def setup_frame_extraction_args() -> dict:
    return {
        'trim_frame_start': config.get_int_value('frame_extraction', 'trim_frame_start'),
        'trim_frame_end': config.get_int_value('frame_extraction', 'trim_frame_end'),
        'temp_frame_format': config.get_str_value('frame_extraction', 'temp_frame_format', 'png'),
        'keep_temp': config.get_bool_value('frame_extraction', 'keep_temp')
    }


def setup_output_creation_args() -> dict:
    available_encoder_set = get_available_encoder_set()

    return {
        'output_image_quality': config.get_int_value('output_creation', 'output_image_quality', '80'),
        'output_image_scale': config.get_float_value('output_creation', 'output_image_scale', '1.0'),
        'output_audio_encoder': config.get_str_value('output_creation', 'output_audio_encoder', get_first(available_encoder_set.get('audio'))),
        'output_audio_quality': config.get_int_value('output_creation', 'output_audio_quality', '80'),
        'output_audio_volume': config.get_int_value('output_creation', 'output_audio_volume', '100'),
        'output_video_encoder': config.get_str_value('output_creation', 'output_video_encoder', get_first(available_encoder_set.get('video'))),
        'output_video_preset': config.get_str_value('output_creation', 'output_video_preset', 'veryfast'),
        'output_video_quality': config.get_int_value('output_creation', 'output_video_quality', '80'),
        'output_video_scale': config.get_float_value('output_creation', 'output_video_scale', '1.0'),
        'output_video_fps': config.get_float_value('output_creation', 'output_video_fps')
    }


def setup_processors_args() -> dict:
    processors_list = config.get_str_list('processors', 'processors', 'face_swapper')
    face_swapper_model = config.get_str_value('processors', 'face_swapper_model', 'hyperswap_1a_256')
    face_swapper_pixel_boost_choices = face_swapper_choices.face_swapper_set.get(face_swapper_model)

    return {
        'processors': processors_list,

        # --- age_modifier ---
        'age_modifier_model': config.get_str_value('processors', 'age_modifier_model', 'styleganex_age'),
        'age_modifier_direction': config.get_int_value('processors', 'age_modifier_direction', '0'),

        # --- deep_swapper ---
        'deep_swapper_model': config.get_str_value('processors', 'deep_swapper_model', 'iperov/elon_musk_224'),
        'deep_swapper_morph': config.get_int_value('processors', 'deep_swapper_morph', '100'),

        # --- background_remover --- (新增)
        'background_remover_model': config.get_str_value('processors', 'background_remover_model', 'rmbg_2.0'),
        'background_remover_color': config.get_int_list('processors', 'background_remover_color', '0 0 0 0'),


        # --- expression_restorer ---
        'expression_restorer_model': config.get_str_value('processors', 'expression_restorer_model', 'live_portrait'),
        'expression_restorer_factor': config.get_int_value('processors', 'expression_restorer_factor', '80'),
        'expression_restorer_areas': config.get_str_list('processors', 'expression_restorer_areas', 'upper-face lower-face'),

        # --- face_debugger --- (新增)
        'face_debugger_items': config.get_str_list('processors', 'face_debugger_items', 'face-landmark-5/68 face-mask'),

        # --- face_editor ---
        'face_editor_model': config.get_str_value('processors', 'face_editor_model', 'live_portrait'),
        'face_editor_eyebrow_direction': config.get_float_value('processors', 'face_editor_eyebrow_direction', '0'),
        'face_editor_eye_gaze_horizontal': config.get_float_value('processors', 'face_editor_eye_gaze_horizontal', '0'),
        'face_editor_eye_gaze_vertical': config.get_float_value('processors', 'face_editor_eye_gaze_vertical', '0'),
        'face_editor_eye_open_ratio': config.get_float_value('processors', 'face_editor_eye_open_ratio', '0'),
        'face_editor_lip_open_ratio': config.get_float_value('processors', 'face_editor_lip_open_ratio', '0'),
        'face_editor_mouth_grim': config.get_float_value('processors', 'face_editor_mouth_grim', '0'),
        'face_editor_mouth_pout': config.get_float_value('processors', 'face_editor_mouth_pout', '0'),
        'face_editor_mouth_purse': config.get_float_value('processors', 'face_editor_mouth_purse', '0'),
        'face_editor_mouth_smile': config.get_float_value('processors', 'face_editor_mouth_smile', '0'),
        'face_editor_mouth_position_horizontal': config.get_float_value('processors', 'face_editor_mouth_position_horizontal', '0'),
        'face_editor_mouth_position_vertical': config.get_float_value('processors', 'face_editor_mouth_position_vertical', '0'),
        'face_editor_head_pitch': config.get_float_value('processors', 'face_editor_head_pitch', '0'),
        'face_editor_head_yaw': config.get_float_value('processors', 'face_editor_head_yaw', '0'),
        'face_editor_head_roll': config.get_float_value('processors', 'face_editor_head_roll', '0'),

        # --- face_enhancer ---
        'face_enhancer_model': config.get_str_value('processors', 'face_enhancer_model', 'gfpgan_1.4'),
        'face_enhancer_blend': config.get_int_value('processors', 'face_enhancer_blend', '80'),
        'face_enhancer_weight': config.get_float_value('processors', 'face_enhancer_weight', '1.0'),

        # --- face_swapper ---
        'face_swapper_model': face_swapper_model,
        'face_swapper_pixel_boost': config.get_str_value('processors', 'face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices)),
        'face_swapper_weight': config.get_float_value('processors', 'face_swapper_weight', '0.5'),
        # --- frame_colorizer ---
        'frame_colorizer_model': config.get_str_value('processors', 'frame_colorizer_model', 'ddcolor'),
        'frame_colorizer_size': config.get_str_value('processors', 'frame_colorizer_size', '256x256'),
        'frame_colorizer_blend': config.get_int_value('processors', 'frame_colorizer_blend', '100'),

        # --- frame_enhancer ---
        'frame_enhancer_model': config.get_str_value('processors', 'frame_enhancer_model', 'span_kendata_x4'),
        #'frame_enhancer_blend': config.get_int_value('processors', 'frame_enhancer_blend', '80'),

        # --- lip_syncer ---
        'lip_syncer_model': config.get_str_value('processors', 'lip_syncer_model', 'wav2lip_gan_96'),
        #'lip_syncer_weight': config.get_float_value('processors', 'lip_syncer_weight', '0.5'),
    }


def setup_execution_args() -> dict:
    available_execution_providers = get_available_execution_providers()
    return {
        'execution_device_ids': config.get_int_list('execution', 'execution_device_ids', '0'),
        'execution_providers': config.get_str_list('execution', 'execution_providers', get_first(available_execution_providers)),
        'execution_thread_count': config.get_int_value('execution', 'execution_thread_count', '8')
    }


def setup_download_providers_args() -> dict:
    return {
        'download_providers': config.get_str_list('download', 'download_providers', ' '.join(facefusion.choices.download_providers))
    }


def setup_memory_args() -> dict:
    return {
        'video_memory_strategy': config.get_str_value('memory', 'video_memory_strategy', 'strict'),
        'system_memory_limit': config.get_int_value('memory', 'system_memory_limit', '0')
    }


def setup_log_level_args() -> dict:
    return {
        'log_level': config.get_str_value('misc', 'log_level', 'info')
    }


def setup_collect_step_args() -> dict:
    """收集所有步骤相关的参数"""
    return {
        **setup_face_detector_args(),
        **setup_face_landmarker_args(),
        **setup_face_selector_args(),
        **setup_face_masker_args(),
        **setup_voice_extractor_args(),
        **setup_frame_extraction_args(),
        **setup_output_creation_args(),
        **setup_processors_args()
    }


def setup_collect_job_args() -> dict:
    """收集所有作业相关的参数"""
    return {
        **setup_execution_args(),
        **setup_download_providers_args(),
        **setup_memory_args(),
        **setup_log_level_args()
    }


def setup_args(config_path='facefusion.ini') -> dict:
    """
    为 headless-run 命令设置参数
    对应: headless-run 命令的父解析器列表
    """
    return {
        **setup_config_path_args(config_path),
        **setup_temp_path_args(),
        **setup_jobs_path_args(),
        **setup_source_paths_args(),
        **setup_target_path_args(),
        **setup_output_path_args(),
        **setup_collect_step_args(),
        **setup_collect_job_args()
    }

def gen_faceswapper_model_args(model_name: str,) -> dict:
    """单独设置 face_swapper_model 参数"""
    face_swapper_pixel_boost_choices = face_swapper_choices.face_swapper_set.get(model_name)
    return {
        'face_swapper_model': model_name,
        'face_swapper_pixel_boost': config.get_str_value('processors', 'face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices))
    }