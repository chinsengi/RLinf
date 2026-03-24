from rlinf.vendor.transformers_pi05.models.gemma.configuration_gemma import GemmaConfig
from rlinf.vendor.transformers_pi05.models.gemma.modeling_gemma import GemmaModel
from rlinf.vendor.transformers_pi05.models.siglip.configuration_siglip import (
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig,
)
from rlinf.vendor.transformers_pi05.models.siglip.modeling_siglip import (
    SiglipModel,
    SiglipTextModel,
    SiglipVisionModel,
)

CONFIG_MAPPING = {
    "gemma": GemmaConfig,
    "siglip": SiglipConfig,
    "siglip_text_model": SiglipTextConfig,
    "siglip_vision_model": SiglipVisionConfig,
}


class AutoConfig:
    @classmethod
    def for_model(cls, model_type, **kwargs):
        return CONFIG_MAPPING[model_type](**kwargs)


class AutoModel:
    @classmethod
    def from_config(cls, config):
        model_type = getattr(config, "model_type", None)
        if model_type == "gemma":
            return GemmaModel(config)
        if model_type == "siglip_vision_model":
            return SiglipVisionModel(config)
        if model_type == "siglip_text_model":
            return SiglipTextModel(config)
        if model_type == "siglip":
            return SiglipModel(config)
        raise ValueError(
            f"Unsupported PI05 vendored AutoModel config type: {model_type}"
        )
