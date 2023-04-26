from conf import default, general, paths
from models.resunet.networks_pos import ResUnetSAR

def get_model(log):
    log.info('Model SAR Resunet sem imagem t0')
    input_depth_0 = general.N_SAR_BANDS + 1
    input_depth_1 = 0
    model_depths = general.RESUNET_DEPTHS
    log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    input_depth = input_depth_0 + input_depth_1
    model = ResUnetSAR(input_depth, model_depths, general.N_CLASSES)

    return model