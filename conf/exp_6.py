from conf import default, general, paths
from models.resunet.networks import ResUnetLF

def get_model(log):
    log.info('Model LF Resunet')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 1
    input_depth_1 = 2*general.N_SAR_BANDS + 1
    model_depths = general.RESUNET_DEPTHS
    log.info(f'Model size: {model_depths}')
    log.info(f'Input depth 0: {input_depth_0}, Input depth 1: {input_depth_1}')
    #input_depth = input_depth_0 + input_depth_1
    model = ResUnetLF(input_depth_0, input_depth_1, model_depths, general.N_CLASSES)

    return model