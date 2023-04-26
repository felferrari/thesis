from conf import general
from models.swin.networks_pos import SwinUnetJF

def get_model(log):
    log.info('Model JF with skip SWINUnet sem imagem t0')
    input_depth_0 = general.N_OPTICAL_BANDS +1
    input_depth_1 = general.N_SAR_BANDS + 1
    #log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    #input_depth = input_depth_0 + input_depth_1
    model = SwinUnetJF(
        input_depth_0 = input_depth_0, 
        input_depth_1 = input_depth_1, 
        img_size = general.PATCH_SIZE,
        base_dim = general.SWIN_BASE_DIM,
        window_size = general.SWIN_WINDOW_SIZE,
        shift_size = general.SWIN_SHIFT_SIZE,
        patch_size = general.SWIN_PATCH_SIZE,
        n_heads = general.SWIN_N_HEADS,
        n_blocks = general.SWIN_N_BLOCKS,
        n_classes = general.N_CLASSES
        )

    return model