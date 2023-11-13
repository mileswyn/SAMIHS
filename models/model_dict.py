from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_samihs.build_sam_ihs import samihs_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == 'MedSAM':
        model = sam_model_registry['vit_b'](checkpoint=None)
    elif modelname == "SAMIHS":
        model = samihs_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
